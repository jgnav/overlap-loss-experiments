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
            values = runner.collect_completed()
            self.assertEqual(values["online_imagenet_cls_knn_top1"], 1.0)
            self.assertEqual(values["online_voc_dense_knn_miou_percent"], 30.0)
            log.close.assert_called_once()

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
            runner.close()


if __name__ == "__main__":
    unittest.main()
