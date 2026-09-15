import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from evaluation.online_probes import (
    OnlineProbeRunner,
    _dense_pixel_metrics,
    _patch_labels,
    _vote_rankings,
    _extract_image_features,
    validate_probe_data,
    immutable_checkpoint_copy,
    knn_classification_metrics,
    probe_due,
    select_fixed_indices,
    select_stratified_indices,
)


class SelectionTest(unittest.TestCase):
    def test_stratified_selection_is_fixed_and_preserves_class_quotas(self):
        targets = torch.tensor([0] * 10 + [1] * 20 + [2] * 30)
        first = select_stratified_indices(targets, 12, 17)
        self.assertEqual(first, select_stratified_indices(targets, 12, 17))
        self.assertEqual(len(first), 12)
        self.assertEqual(torch.bincount(targets[first], minlength=3).tolist(), [2, 4, 6])

    def test_fixed_selection_is_seeded(self):
        self.assertEqual(select_fixed_indices(10, 4, 1), select_fixed_indices(10, 4, 1))
        self.assertNotEqual(select_fixed_indices(10, 4, 1), select_fixed_indices(10, 4, 2))
        self.assertTrue(probe_due(10, 10))
        self.assertFalse(probe_due(11, 10))


class KNNTest(unittest.TestCase):
    def test_batched_votes_match_reference_with_ties(self):
        generator = torch.Generator().manual_seed(9)
        labels = torch.randint(0, 7, (300, 20), generator=generator) * 2
        values = torch.randint(-2, 3, labels.shape, generator=generator).float()
        classes = torch.arange(7) * 2
        expected = []
        for row, similarities in zip(labels, values):
            expected.append(sorted(classes.tolist(), key=lambda label: (
                int((row == label).sum()), float(similarities[row == label].sum()), -label
            ), reverse=True)[:5])
        self.assertEqual(_vote_rankings(labels, values, classes).tolist(), expected)

    def test_cosine_knn_reports_top1_and_top5_without_pairwise_bank_matrix(self):
        train = torch.eye(6)
        labels = torch.arange(6)
        query = train[[0, 4]]
        result = knn_classification_metrics(train, labels, query, torch.tensor([0, 4]), k=1)
        self.assertEqual(result, {"top1": 100.0, "top5": 100.0})


class DenseMetricTest(unittest.TestCase):
    def test_patch_labels_majority_and_ignore(self):
        labels = torch.full((1, 256, 256), 255, dtype=torch.long)
        labels[:, :16, :16] = 3
        labels[:, :16, 16:32] = 4
        patch_labels = _patch_labels(labels)
        self.assertEqual(patch_labels[:2].tolist(), [3, 4])
        self.assertTrue((patch_labels[2:] == 255).all())

    def test_dense_scores_ignore_pixels(self):
        target = torch.zeros(1, 256, 256, dtype=torch.long)
        target[:, :16, :16] = 255
        predicted = torch.zeros(256, dtype=torch.long)
        result = _dense_pixel_metrics(predicted, target)
        self.assertEqual(result["pixel_accuracy"], 1.0)
        self.assertEqual(result["miou"], 1.0)


class RunnerTest(unittest.TestCase):
    def _args(self, root):
        return SimpleNamespace(
            online_probes_enabled=True,
            online_probe_frequency=10,
            output_dir=str(root),
            online_probe_datasets_root="dataset",
            arch="auto",
            seed=0,
            online_probe_imagenet_train_size=10000,
            online_probe_imagenet_val_size=5000,
            online_probe_voc_train_size=400,
            online_probe_voc_val_size=200,
            online_probe_k=20,
            online_probe_batch_size=256,
            online_probe_num_workers=0,
            online_probe_max_concurrent_jobs=2,
            online_probe_gpu=None,
        )

    def test_snapshot_is_immutable_and_completed_metrics_are_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint.pth"
            source.write_bytes(b"checkpoint bytes")
            copy = immutable_checkpoint_copy(source, root / "copy.pth")
            self.assertEqual(copy.read_bytes(), source.read_bytes())

            runner = OnlineProbeRunner(self._args(root), repository_root=root)
            process = mock.Mock()
            process.poll.return_value = 0
            process.pid = 12
            result_path = root / "result.json"
            result_path.write_text(json.dumps({
                "status": "completed", "epoch": 10,
                "imagenet_cls_knn": {"top1": 1, "top5": 2},
                "voc_dense_knn": {
                    "miou": .3, "miou_percent": 30,
                    "pixel_accuracy": .4, "pixel_accuracy_percent": 40,
                },
            }))
            log = mock.Mock()
            runner.processes = [(process, log)]
            runner.submitted_results = [(10, result_path)]
            values, = runner.collect_completed()
            self.assertEqual(values["online_imagenet_cls_knn_top1"], 1.0)
            self.assertEqual(values["online_voc_dense_knn_miou_percent"], 30.0)
            log.close.assert_called_once()
            self.assertEqual(runner.collect_completed(), [])

    def test_multiple_results_and_failures_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = OnlineProbeRunner(self._args(root), repository_root=root)
            for epoch in (10, 20):
                path = root / f"epoch{epoch}.json"
                path.write_text(json.dumps({"status": "failed", "epoch": epoch,
                                            "error": "missing validation data"}))
                runner.submitted_results.append((epoch, path))
            self.assertEqual(runner.collect_completed(), [
                {"online_probe_epoch": 10, "online_probe_success": 0},
                {"online_probe_epoch": 20, "online_probe_success": 0},
            ])
            self.assertEqual(runner.collect_completed(), [])

    def test_dead_worker_without_json_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = OnlineProbeRunner(self._args(root), repository_root=root)
            process = mock.Mock(pid=12, returncode=-9)
            process.poll.return_value = -9
            path = root / "epoch0010.json"
            runner.processes = [(process, mock.Mock())]
            runner.process_results[12] = (10, path)
            runner.submitted_results = [(10, path)]
            record, = runner.collect_completed()
            self.assertEqual(record["online_probe_success"], 0)
            self.assertIn("-9", json.loads(path.read_text())["error"])

    def test_submit_skips_non_due_epochs_and_launches_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint.pth"
            source.write_bytes(b"checkpoint bytes")
            runner = OnlineProbeRunner(self._args(root), repository_root=root)
            with mock.patch("evaluation.online_probes.subprocess.Popen") as launch:
                launch.return_value.poll.return_value = None
                launch.return_value.pid = 7
                self.assertIsNone(runner.submit(9, source))
                submitted = runner.submit(10, source)
            self.assertEqual(submitted["checkpoint"].read_bytes(), b"checkpoint bytes")
            self.assertTrue(submitted["checkpoint"].name == "teacher_epoch0010.pth")
            launch.assert_called_once()
            self.assertIn("evaluation.online_probes", launch.call_args.args[0])
            command = launch.call_args.args[0]
            for option in ("--checkpoint", "--datasets-root", "--output"):
                self.assertTrue(Path(command[command.index(option) + 1]).is_absolute())
            runner.close()


class FeatureAndPathTest(unittest.TestCase):
    def test_final_cls_token_and_deterministic_extraction(self):
        model = mock.Mock()
        model.get_intermediate_layers.return_value = [torch.tensor([
            [[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]
        ])]
        dataset = torch.utils.data.TensorDataset(torch.zeros(2, 3, 224, 224), torch.tensor([0, 1]))
        features, labels = _extract_image_features(model, dataset, 2, 0, device=torch.device("cpu"))
        self.assertEqual(features.tolist(), [[1., 2.], [5., 6.]])
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(model.get_intermediate_layers.call_args.kwargs, {"n": 1})

    def test_missing_validation_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "imagenet" / "train").mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "train/ and val/"):
                validate_probe_data(directory)

    def test_logging_uses_checkpoint_epoch_and_keeps_all_records(self):
        from train import log_online_probe_records
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "online_probes").mkdir()
            runner, writer, wandb = mock.Mock(), mock.Mock(), mock.Mock()
            runner.collect_completed.return_value = [
                {"online_probe_epoch": 10, "online_probe_success": 1},
                {"online_probe_epoch": 20, "online_probe_success": 0},
            ]
            log_online_probe_records(runner, directory, writer, wandb)
            self.assertEqual(wandb.log.call_count, 2)
            self.assertEqual(wandb.log.call_args.args[0]["train/online_probe_epoch"], 20)
            writer.add_scalar.assert_any_call("online_probe_success", 1, 10)
            records = (Path(directory) / "online_probes" / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
