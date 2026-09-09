import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from evaluation.prepare_voc_manifest import VOC_CLASSES, build_voc2012_manifest, prepare_voc2012_manifest
from evaluation.utils import classification, dense, orchestrator
from evaluation.utils.classification_data import read_multilabel_manifest
from evaluation.utils.common import checkpoint_fingerprint, evaluation_identity
from evaluation.utils.datasets import make_pascal_voc, segmentation_manifest
from model.vision_transformer import VisionTransformer


class _IdentityFeatures(torch.nn.Module):
    def get_intermediate_layers(self, images, n=1):
        # Two identical tokens make ViT-B feature pooling concatenate x with x.
        return [images[:, None, :].expand(-1, 2, -1)]


def _distributed_probe_worker(rank, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(4)
        head = torch.nn.Linear(3, 2)
        reference = copy.deepcopy(head)
        ddp = torch.nn.parallel.DistributedDataParallel(head)
        inputs = torch.tensor([[0.2, 0.4, 0.6], [-0.1, 0.1, 0.3]])
        targets = torch.tensor([[1.0, -1.0], [0.0, 1.0]])
        loss = classification.classification_loss(ddp(inputs[rank:rank + 1]), targets[rank:rank + 1], True)
        loss.backward()
        valid = targets >= 0
        expected = F.binary_cross_entropy_with_logits(reference(inputs)[valid], targets[valid])
        expected.backward()
        torch.testing.assert_close(head.weight.grad, reference.weight.grad)
        torch.testing.assert_close(head.bias.grad, reference.bias.grad)

        # Three validation samples cannot be evenly divided over two ranks.
        # The distributed result must equal AP on all three unique images.
        values = torch.tensor([[0.9, 0.1], [0.8, 0.7], [0.1, 0.8]])
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        dataset = TensorDataset(values, labels)
        dataset.classes = ["cat", "dog"]
        projection = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
        result = classification.evaluate(_IdentityFeatures(), projection, dataset, "vit_base", True, "cpu", rank, 2, 0)
        if rank == 0:
            expected = classification.multilabel_metrics(labels.numpy(), values.numpy(), dataset.classes)
            assert result == expected, (result, expected)
        else:
            assert result is None

        # Also exercise a rank with no local validation samples.
        dataset = TensorDataset(values[:1], labels[:1])
        dataset.classes = ["cat", "dog"]
        result = classification.evaluate(_IdentityFeatures(), projection, dataset, "vit_base", True, "cpu", rank, 2, 0)
        if rank == 0:
            assert result["map"] == 0.5
    finally:
        dist.destroy_process_group()


class VOCConstructionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / "VOCdevkit/VOC2012"
        self.augmented = self.root / "benchmark_RELEASE/dataset"
        splits = self.original / "ImageSets/Segmentation"
        splits.mkdir(parents=True)
        self.augmented.mkdir(parents=True)
        (splits / "train.txt").write_text("a\nd\n")
        (splits / "val.txt").write_text("heldout\n")
        (self.augmented / "train.txt").write_text("a\nb\nheldout\n")
        (self.augmented / "val.txt").write_text("c\nd\n")
        masks = self.original / "SegmentationClass"
        masks.mkdir()
        for item in ("a", "d", "heldout"):
            Image.new("L", (8, 8)).save(masks / f"{item}.png")

    def test_original_splits_ignore_sbd_and_have_zero_overlap(self):
        dataset = make_pascal_voc(self.root, "train")
        self.assertEqual([p.stem for p in dataset.images], ["a", "d"])
        self.assertEqual(dataset.targets[0], self.original / "SegmentationClass/a.png")
        self.assertEqual(dataset.targets[1], self.original / "SegmentationClass/d.png")
        meta = segmentation_manifest(dataset)
        self.assertEqual(meta["construction"], "voc2012_original_segmentation_v1")
        self.assertEqual(meta["repeated_image_entries"], 0)
        self.assertEqual(meta["official_val_overlap_unique_ids"], 0)
        self.assertEqual(meta["unique_image_ids"], 2)
        before = meta["ordered_pairs_sha256"]
        dataset.targets[1] = dataset.targets[0]
        self.assertNotEqual(segmentation_manifest(dataset)["ordered_pairs_sha256"], before)

    def test_works_without_sbd_and_preserves_official_order(self):
        for path in self.augmented.iterdir():
            path.unlink()
        self.augmented.rmdir()
        (self.original / "ImageSets/Segmentation/train.txt").write_text("d\na\n")
        dataset = make_pascal_voc(self.root, "train")
        self.assertEqual([p.stem for p in dataset.images], ["d", "a"])

    def test_rejects_trainaug_instead_of_silently_changing_its_meaning(self):
        with self.assertRaisesRegex(ValueError, "original train or val"):
            make_pascal_voc(self.root, "trainaug")

    def test_rejects_duplicate_train_and_cross_split_overlap(self):
        path = self.original / "ImageSets/Segmentation/train.txt"
        path.write_text("a\na\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            make_pascal_voc(self.root, "train")
        path.write_text("a\nheldout\n")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            make_pascal_voc(self.root, "train")

    def test_official_validation_uses_only_original_masks(self):
        dataset = make_pascal_voc(self.root, "val")
        self.assertEqual(dataset.images, [self.original / "JPEGImages/heldout.jpg"])
        self.assertEqual(dataset.targets, [self.original / "SegmentationClass/heldout.png"])

    def test_rejects_duplicate_official_validation_ids(self):
        (self.original / "ImageSets/Segmentation/val.txt").write_text("heldout\nheldout\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            make_pascal_voc(self.root, "val")

    def test_internal_holdout_is_image_disjoint_and_seeded(self):
        (self.original / "ImageSets/Segmentation/train.txt").write_text("\n".join(["a", "b", "c", "d"] + [f"extra{i}" for i in range(20)]))
        sets = dense._build_dense_datasets("pascal_voc", self.root, 0)
        train_ids = {sets["train"].dataset.images[i].stem for i in sets["train"].indices}
        holdout_ids = {sets["val"].dataset.images[i].stem for i in sets["val"].indices}
        test_ids = {p.stem for p in sets["test"].images}
        self.assertEqual(len(train_ids), 22)
        self.assertEqual(len(holdout_ids), 2)
        self.assertFalse(train_ids & holdout_ids or train_ids & test_ids or holdout_ids & test_ids)
        repeated = dense._build_dense_datasets("pascal_voc", self.root, 0)
        self.assertEqual(sets["val"].indices, repeated["val"].indices)


class VOCClassificationPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.voc = self.root / "pascal_voc/VOCdevkit/VOC2012"
        self.main = self.voc / "ImageSets/Main"
        self.main.mkdir(parents=True)
        images = self.voc / "JPEGImages"
        images.mkdir()
        for item in ("a", "b", "c", "d", "difficult"):
            Image.new("RGB", (8, 8)).save(images / f"{item}.jpg")
        for split, positive, negative in (("train", "a", "b"), ("val", "c", "d")):
            ids = [positive, negative] + (["difficult"] if split == "val" else [])
            (self.main / f"{split}.txt").write_text("\n".join(ids) + "\n")
            for index, class_name in enumerate(VOC_CLASSES):
                # Deliberately reverse annotation rows: alignment must use IDs.
                rows = f"{negative} -1\n{positive} 1\n"
                if split == "val":
                    rows += f"difficult {1 if index == 0 else 0}\n"
                (self.main / f"{class_name}_{split}.txt").write_text(rows)

    def test_prepares_valid_reproducible_classification_manifest(self):
        result = prepare_voc2012_manifest(self.root)
        path = Path(result["manifest"])
        payload = path.read_bytes()
        samples, classes, metadata = read_multilabel_manifest(path, self.root, "pascal_voc", 20)
        self.assertEqual(classes, list(VOC_CLASSES))
        self.assertEqual(metadata["split_sizes"], {"train": 2, "val": 3})
        self.assertEqual(samples["train"][0][1], [1] * 20)
        self.assertEqual(samples["train"][1][1], [0] * 20)
        self.assertEqual(samples["val"][2][1], [1] + [-1] * 19)
        manifest = json.loads(payload)
        self.assertEqual(manifest["splits"]["val"][2]["labels"], [1] + [None] * 19)
        self.assertFalse(Path(manifest["splits"]["train"][0]["image"]).is_absolute())
        self.assertEqual(len(manifest["source_files_sha256"]), 42)
        prepare_voc2012_manifest(self.root)
        self.assertEqual(path.read_bytes(), payload)

    def test_refuses_to_replace_an_existing_different_manifest(self):
        output = self.root / "existing.json"
        output.write_text("existing data")
        with self.assertRaises(FileExistsError):
            prepare_voc2012_manifest(self.root, output=output)
        self.assertEqual(output.read_text(), "existing data")

    def test_rejects_native_label_and_annotation_id_errors(self):
        path = self.main / "aeroplane_train.txt"
        for text in ("a 2\nb -1\n", "a 1\na -1\n", "a 1\nunknown -1\n", "a 1\n"):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    build_voc2012_manifest(self.root)

    def test_rejects_overlapping_or_duplicate_split_ids(self):
        for text in ("c\na\n", "c\nc\n"):
            (self.main / "val.txt").write_text(text)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "overlap"):
                build_voc2012_manifest(self.root)

    def test_missing_images_fail_before_publishing_manifest(self):
        (self.voc / "JPEGImages/a.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_voc2012_manifest(self.root)
        self.assertFalse((self.root / "evaluation_manifests/pascal_voc.json").exists())


class SegmentationResolutionTest(unittest.TestCase):
    def test_256_resolution_produces_256_tokens_with_aligned_labels(self):
        image_transform, target_transform = dense._dense_transforms()
        image = image_transform(Image.new("RGB", (400, 200)))
        # Each patch has a unique row-major label. Patchification must retain the
        # feature grid's order without averaging labels or mixing adjacent patches.
        target = np.arange(256, dtype=np.uint8).reshape(16, 16).repeat(16, 0).repeat(16, 1)
        labels = target_transform(Image.fromarray(target))
        # Start with a 224-pretrained positional grid; evaluation must interpolate.
        model = VisionTransformer(img_size=[224], patch_size=16, embed_dim=12, depth=1, num_heads=3).eval()
        with torch.no_grad():
            tokens = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0][:, 1:]
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(tokens.shape, (1, 256, 12))
        patches = dense._patchify_labels(labels[None], 16, 16)
        self.assertTrue(torch.equal(patches, torch.arange(256, dtype=torch.uint8)[:, None].expand(-1, 256)))

    def test_orchestrator_does_not_supply_legacy_feature_caches(self):
        self.assertTrue(all(cache is None for _, _, cache in orchestrator.EVALUATIONS))


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "toy.json"
        for name in ("a", "b", "c", "d"):
            Image.new("RGB", (8, 8)).save(self.root / f"{name}.png")
        self.manifest = {
            "dataset": "toy", "source": "Test fixture", "classes": ["cat", "dog"],
            "splits": {
                "train": [{"id": "a", "image": "a.png", "labels": [1, 0]},
                          {"id": "b", "image": "b.png", "labels": [0, 1]}],
                "val": [{"id": "c", "image": "c.png", "labels": [1, None]},
                        {"id": "d", "image": "d.png", "labels": [0, 1]}],
            },
        }

    def read(self, manifest=None):
        self.path.write_text(json.dumps(manifest or self.manifest))
        return read_multilabel_manifest(self.path, self.root, "toy", 2)

    def test_preserves_vocabulary_and_ignored_labels(self):
        samples, classes, metadata = self.read()
        self.assertEqual(classes, ["cat", "dog"])
        self.assertEqual(samples["val"][0][1], [1, -1])
        self.assertEqual(metadata["split_sizes"], {"train": 2, "val": 2})

    def test_rejects_overlap_by_id_even_with_different_paths(self):
        self.manifest["splits"]["val"][0]["id"] = "a"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.read()

    def test_rejects_path_alias_overlap(self):
        self.manifest["splits"]["val"][0]["image"] = "./a.png"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.read()

    def test_rejects_file_symlink_overlap(self):
        (self.root / "alias.png").symlink_to(self.root / "a.png")
        self.manifest["splits"]["val"][0]["image"] = "alias.png"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.read()

    def test_rejects_parent_symlink_overlap(self):
        (self.root / "alias").symlink_to(self.root, target_is_directory=True)
        self.manifest["splits"]["val"][0]["image"] = "alias/a.png"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.read()

    def test_rejects_missing_files_and_directories(self):
        for name in ("missing.png", "."):
            self.manifest["splits"]["val"][0]["image"] = name
            with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                self.read()

    def test_rejects_raw_voc_labels_and_invalid_vocabularies(self):
        for labels in ([-1, 1], [2, 1], [True, 0], [1], [None, None]):
            manifest = copy.deepcopy(self.manifest)
            manifest["splits"]["val"][0]["labels"] = labels
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.read(manifest)
        self.manifest["classes"] = ["cat", "cat"]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.read()

    def test_rejects_fully_unknown_validation_class_before_training(self):
        for row in self.manifest["splits"]["val"]:
            row["labels"][1] = None
        with self.assertRaisesRegex(ValueError, "validation needs known labels"):
            self.read()


class ClassificationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_feature_pooling_for_small_and_large_backbones(self):
        layers = [torch.arange(18).reshape(2, 3, 3).float() + 100 * i for i in range(4)]
        backbone = mock.Mock()
        backbone.get_intermediate_layers.return_value = layers
        features = classification.classification_features(backbone, torch.empty(0), "vit_small")
        torch.testing.assert_close(features, torch.cat([x[:, 0] for x in layers], dim=1))
        backbone.get_intermediate_layers.return_value = [layers[-1]]
        features = classification.classification_features(backbone, torch.empty(0), "vit_base")
        torch.testing.assert_close(features, torch.cat([layers[-1][:, 0], layers[-1][:, 1:].mean(1)], dim=1))

    def test_unknown_labels_do_not_contribute_loss_or_gradient(self):
        logits = torch.tensor([[2.0, -3.0], [-1.0, 4.0]], requires_grad=True)
        labels = torch.tensor([[1.0, -1.0], [0.0, 1.0]])
        loss = classification.classification_loss(logits, labels, True)
        expected = F.binary_cross_entropy_with_logits(logits[labels >= 0], labels[labels >= 0])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 1].item(), 0)

    def test_map_is_global_per_class_and_masks_unknown_labels(self):
        labels = np.array([[1, 0], [0, 1], [1, -1], [0, 0]])
        scores = np.array([[0.9, 0.8], [0.8, 0.9], [0.7, 100.0], [0.1, 0.1]])
        metrics = classification.multilabel_metrics(labels, scores, ["cat", "dog"])
        self.assertAlmostEqual(metrics["average_precision_by_class"]["cat"], 5 / 6)
        self.assertAlmostEqual(metrics["average_precision_by_class"]["dog"], 1)
        self.assertAlmostEqual(metrics["map"], 11 / 12)
        zero = classification.multilabel_metrics(np.array([[0], [0]]), np.array([[1], [2]]), ["absent"])
        self.assertEqual(zero["map"], 0)
        self.assertEqual(zero["classes_without_validation_positives"], ["absent"])

    def test_probe_training_changes_head_but_not_backbone(self):
        for multilabel in (False, True):
            with self.subTest(multilabel=multilabel):
                torch.manual_seed(1)
                backbone = VisionTransformer(img_size=[32], patch_size=16, embed_dim=12, depth=4, num_heads=3).eval()
                backbone.requires_grad_(False)
                head = classification.linear_head(backbone, "vit_small", 2)
                images = torch.randn(8, 3, 32, 32)
                labels = torch.arange(8) % 2
                if multilabel:
                    labels = F.one_hot(labels, num_classes=2).float()
                    labels[0, 1] = -1
                loader = DataLoader(TensorDataset(images, labels), batch_size=4)
                before = {key: value.clone() for key, value in backbone.state_dict().items()}
                head_before = head.weight.detach().clone()
                optimizer = torch.optim.SGD(head.parameters(), lr=0.001, momentum=0.9)
                loss = classification.train_epoch(backbone, head, optimizer, loader, "vit_small", multilabel, "cpu", 0, 1)
                self.assertTrue(np.isfinite(loss))
                self.assertFalse(torch.equal(head.weight, head_before))
                self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
                for key, value in backbone.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                self.assertFalse(backbone.training)

    def test_distributed_gradients_and_uneven_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(
                _distributed_probe_worker, args=(str(Path(directory) / "rendezvous"),),
                nprocs=2, join=True, start_method="fork",
            )


class OrchestrationTest(unittest.TestCase):
    def test_table_keeps_segmentation_and_multilabel_tasks_separate(self):
        results = {
            "pascal_voc_linear": {"metrics": {"miou": 0.6}},
            "pascal_voc_multilabel": {"dataset": "PASCAL VOC", "metrics": {"map": 0.9}},
            "imagenet_linear": {"metrics": {"top1": 77.9}},
        }
        table = orchestrator._result_table(results)
        self.assertEqual(len(table), 3)
        self.assertEqual({row["task"] for row in table}, {"semantic_segmentation", "multilabel_classification", "multiclass_classification"})
        self.assertEqual(len(orchestrator.EVALUATIONS), 15)

    def test_previous_results_and_changed_seed_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"fixture")
            args = SimpleNamespace(checkpoint=checkpoint, checkpoint_key="teacher", seed=0, arch="auto", datasets_root=root)
            result = {
                "status": "completed", "evaluation": "pascal_voc_linear",
                "model": {"checkpoint_fingerprint": checkpoint_fingerprint(checkpoint), "checkpoint_key": "teacher"},
            }
            path = root / "result.json"
            path.write_text(json.dumps(result))
            self.assertIsNone(orchestrator._load_completed_result(path, args, "pascal_voc_linear"))
            result["evaluation_identity"] = evaluation_identity(args)
            path.write_text(json.dumps(result))
            self.assertIsNotNone(orchestrator._load_completed_result(path, args, "pascal_voc_linear"))
            args.seed = 1
            self.assertIsNone(orchestrator._load_completed_result(path, args, "pascal_voc_linear"))


if __name__ == "__main__":
    unittest.main()
