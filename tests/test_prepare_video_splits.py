"""Check that fallback MOSE splits preserve the published evaluation population."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from evaluation.prepare_video_splits import mose_manifest


class MOSESplitTest(unittest.TestCase):
    def test_original_mose_partition_is_disjoint_complete_and_reproducible(self):
        names = [f"video{i:04d}" for i in range(1507)]
        with patch("evaluation.prepare_video_splits.video_ids", return_value=names):
            first = mose_manifest(Path("/datasets"), seed=0)
            repeated = mose_manifest(Path("/datasets"), seed=0)
            different = mose_manifest(Path("/datasets"), seed=1)
        selection, evaluation = first["splits"].values()
        self.assertEqual((len(selection), len(evaluation)), (1206, 301))
        self.assertFalse(set(selection) & set(evaluation))
        self.assertEqual(set(selection) | set(evaluation), set(names))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first["splits"], different["splits"])
        self.assertEqual(first["dataset_release"], "2023")
        self.assertTrue(first["paper_split_sizes_matched"])
        self.assertFalse(first["author_split_verified"])
        self.assertEqual(first["image_root"], "MOSE2023/train/JPEGImages")

    def test_wrong_release_population_or_missing_annotation_videos_rejected(self):
        with patch("evaluation.prepare_video_splits.video_ids", return_value=["a"]):
            with self.assertRaisesRegex(ValueError, "1507"):
                mose_manifest(Path("/datasets"))
        names = [f"video{i:04d}" for i in range(1507)]
        with patch("evaluation.prepare_video_splits.video_ids", side_effect=[names, names[:-1]]):
            with self.assertRaisesRegex(ValueError, "annotation video IDs differ"):
                mose_manifest(Path("/datasets"))

    def test_official_catalog_allows_archived_validation_but_requires_extracted_test(self):
        names = [f"video{i:04d}" for i in range(1507)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("evaluation.prepare_video_splits.video_ids", return_value=names):
                expected = mose_manifest(root)
            folder = root / "MOSE2023"
            folder.mkdir()
            (folder / "meta_train.json").write_text(json.dumps({"videos": dict.fromkeys(names, {})}))
            test_ids = expected["splits"]["evaluation"]
            for name in test_ids:
                (folder / "train/JPEGImages" / name).mkdir(parents=True)
                (folder / "train/Annotations" / name).mkdir(parents=True)
            actual = mose_manifest(root)
            self.assertEqual(actual["splits"], expected["splits"])
            self.assertEqual(actual["population_source"], "official meta_train.json")
            (folder / "train/JPEGImages" / test_ids[0]).rmdir()
            with self.assertRaisesRegex(ValueError, "test videos are missing"):
                mose_manifest(root)


if __name__ == "__main__":
    unittest.main()
