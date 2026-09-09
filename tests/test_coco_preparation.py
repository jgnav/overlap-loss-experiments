import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.prepare_coco_manifest import build_coco2017_manifest, prepare_coco2017_manifest
from evaluation.utils.classification_data import read_multilabel_manifest


class CocoPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.coco = self.root / "coco"
        (self.coco / "annotations").mkdir(parents=True)
        # Noncontiguous IDs and reversed categories catch an ID-minus-one bug.
        self.categories = [{"id": 2 * i + 1, "name": f"class{i}"} for i in range(80)]
        self.data = {}
        for split, first in (("train", 1), ("val", 3)):
            images = [{"id": item, "file_name": f"{item:012d}.jpg"} for item in (first, first + 1)]
            directory = self.coco / "images" / f"{split}2017"
            directory.mkdir(parents=True)
            for image in images:
                (directory / image["file_name"]).touch()
            self.data[split] = {
                "images": images,
                "categories": copy.deepcopy(list(reversed(self.categories))),
                "annotations": [
                    {"id": i, "image_id": first, "category_id": category["id"], "iscrowd": int(i == 0)}
                    for i, category in enumerate(self.categories)
                ],
            }
        self.save()

    def save(self):
        for split, data in self.data.items():
            (self.coco / "annotations" / f"instances_{split}2017.json").write_text(json.dumps(data))

    def test_sorted_categories_crowds_negatives_and_all_images(self):
        manifest = build_coco2017_manifest(self.root)
        self.assertEqual(manifest["category_ids"], [category["id"] for category in self.categories])
        self.assertEqual(manifest["classes"], [category["name"] for category in self.categories])
        for split in ("train", "val"):
            self.assertEqual(len(manifest["splits"][split]), 2)
            self.assertEqual(manifest["splits"][split][0]["labels"], [1] * 80)
            self.assertEqual(manifest["splits"][split][1]["labels"], [0] * 80)
            self.assertEqual(manifest["split_statistics"][split]["crowd_instances_included"], 1)
            self.assertEqual(manifest["split_statistics"][split]["images_without_annotations_retained"], 1)
        self.assertEqual(len(manifest["source_files_sha256"]), 2)

    def test_published_manifest_passes_evaluator_and_is_reproducible(self):
        metadata = prepare_coco2017_manifest(self.root)
        output = Path(metadata["manifest"])
        before = output.read_bytes()
        samples, classes, result = read_multilabel_manifest(output, self.root, "coco", 80)
        self.assertEqual(result["split_sizes"], {"train": 2, "val": 2})
        self.assertEqual(len(classes), 80)
        self.assertEqual(samples["train"][0][1], [1] * 80)
        prepare_coco2017_manifest(self.root)
        self.assertEqual(output.read_bytes(), before)

    def test_rejects_duplicate_images_and_train_val_overlap(self):
        original = copy.deepcopy(self.data)
        for split in ("train", "val"):
            self.data = copy.deepcopy(original)
            self.data[split]["images"][1]["id"] = 1
            self.save()
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "overlap"):
                build_coco2017_manifest(self.root)

    def test_rejects_unknown_annotation_references(self):
        original = copy.deepcopy(self.data)
        for field in ("image_id", "category_id"):
            self.data = copy.deepcopy(original)
            self.data["train"]["annotations"][0][field] = 99999
            self.save()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unknown"):
                build_coco2017_manifest(self.root)

    def test_rejects_mismatched_categories_and_path_traversal(self):
        self.data["val"]["categories"][0]["name"] = "different"
        self.save()
        with self.assertRaisesRegex(ValueError, "vocabularies differ"):
            build_coco2017_manifest(self.root)
        self.data["val"]["categories"] = copy.deepcopy(self.data["train"]["categories"])
        self.data["train"]["images"][0]["file_name"] = "../outside.jpg"
        self.save()
        with self.assertRaisesRegex(ValueError, "filename"):
            build_coco2017_manifest(self.root)

    def test_missing_images_fail_without_publishing(self):
        (self.coco / "images/train2017/000000000001.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_coco2017_manifest(self.root)
        self.assertFalse((self.root / "evaluation_manifests/coco.json").exists())

    def test_does_not_overwrite_existing_different_manifest(self):
        path = self.root / "existing.json"
        path.write_text("existing data")
        with self.assertRaises(FileExistsError):
            prepare_coco2017_manifest(self.root, output=path)
        self.assertEqual(path.read_text(), "existing data")

    def test_respects_concurrent_publication_lock(self):
        directory = self.root / "evaluation_manifests"
        directory.mkdir()
        lock = directory / ".coco.json.publish-lock"
        lock.mkdir()
        with self.assertRaisesRegex(FileExistsError, "publication lock"):
            prepare_coco2017_manifest(self.root)
        self.assertTrue(lock.is_dir())
        self.assertFalse((directory / "coco.json").exists())


if __name__ == "__main__":
    unittest.main()
