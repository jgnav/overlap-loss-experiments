"""MOSEv2 export must preserve labels and never score initialization masks."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from evaluation.utils import video_segmentation as vos
from evaluation.utils.common import checkpoint_fingerprint
from evaluation.utils.orchestrator import _load_completed_result, _result_table


class PositionBackbone:
    def get_intermediate_layers(self, images, n=4):
        tokens = torch.cat((torch.zeros(1, 1, 4), torch.eye(4)[None]), dim=1)
        return [tokens.to(images.device) for _ in range(n)]

    def to(self, device):
        return self

    def eval(self):
        return self


class MosePredictionExportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "MOSEv2"
        self.images = self.dataset / "valid/JPEGImages/example"
        self.masks = self.dataset / "valid/Annotations/example"
        self.images.mkdir(parents=True)
        self.masks.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (40, 32), color="gray").save(self.images / f"{index:05d}.jpg")
        self.initial = np.zeros((32, 40), dtype=np.uint8)
        self.initial[:16, :20] = 7
        self.palette = [value for value in range(256) for _ in range(3)]
        vos._save_mask(self.masks / "00000.png", self.initial, self.palette)
        self.video_metadata = {"frames": [f"{index:05d}.jpg" for index in range(3)],
                               "length": 3, "height": 32, "width": 40, "objects": [7]}
        (self.dataset / "meta_valid.json").write_text(
            json.dumps({"videos": {"example": self.video_metadata}}))
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"checkpoint fixture")
        self.model_metadata = {"patch_size": 240, "num_register_tokens": 0,
                               "checkpoint_key": "teacher",
                               "checkpoint_fingerprint": checkpoint_fingerprint(self.checkpoint)}

    def test_preflight_accepts_initialization_only_and_checks_complete_rgb_split(self):
        vos.preflight_masks(self.root, "mose")
        (self.images / "00002.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "RGB frames"):
            vos.preflight_masks(self.root, "mose")

    def test_preflight_requires_initialization_mask(self):
        (self.masks / "00000.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Initial validation mask"):
            vos.preflight_masks(self.root, "mose")

    def test_export_preserves_native_size_palette_ids_and_ignores_later_annotations(self):
        # A later annotation with an unknown ID must never be used in export mode.
        Image.fromarray(np.full_like(self.initial, 99)).save(self.masks / "00001.png")
        predictions = self.root / "predictions/example"
        with mock.patch.object(vos, "_score_frame", side_effect=AssertionError("must not score")):
            scores, details = vos._score_video(
                PositionBackbone(), self.model_metadata, sorted(self.images.iterdir()),
                self.masks, torch.device("cpu"), "mose", prediction_folder=predictions)
        self.assertEqual(scores, {})
        self.assertEqual(details["scored_frames"], 0)
        self.assertEqual(details["exported_frames"], 3)
        self.assertEqual(details["objects"], [7])
        self.assertEqual(sorted(path.name for path in predictions.iterdir()),
                         ["00000.png", "00001.png", "00002.png"])
        for path in predictions.iterdir():
            with Image.open(path) as mask:
                self.assertEqual(mask.mode, "P")
                self.assertEqual(mask.size, (40, 32))
                self.assertEqual(mask.getpalette(), self.palette)
                self.assertTrue(set(np.unique(np.asarray(mask))) <= {0, 7})
        np.testing.assert_array_equal(vos._annotation(predictions / "00000.png"), self.initial)

    def run_worker(self, dataset_name):
        output = self.root / "output"
        result = output / f"{dataset_name}_vos.json"
        argv = ["vos", str(self.checkpoint), "--datasets-root", str(self.root),
                "--output-dir", str(output), "--result-json", str(result)]
        cpu = torch.device("cpu")
        with (mock.patch("sys.argv", argv),
              mock.patch.object(vos.torch.cuda, "is_available", return_value=True),
              mock.patch.object(vos.torch, "device", return_value=cpu),
              mock.patch.object(vos, "load_backbone", return_value=(PositionBackbone(), self.model_metadata))):
            vos.main(dataset_name)
        return result, json.loads(result.read_text())

    def test_worker_writes_submission_zip_and_pending_metrics_then_validates_resume(self):
        with mock.patch.object(vos, "_score_frame", side_effect=AssertionError("must not score")):
            result_path, result = self.run_worker("mose")
        self.assertEqual(result["dataset"], "MOSEv2 val")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics_status"], "pending_external_evaluation")
        self.assertEqual(result["metrics"], {"j_and_f": None, "j_mean": None, "f_mean": None})
        self.assertEqual(result["prediction_export"]["frames"], 3)
        self.assertEqual(result["videos"]["example"]["object_ids"], [7])
        archive = Path(result["prediction_export"]["archive"])
        with zipfile.ZipFile(archive) as submission:
            self.assertEqual(sorted(submission.namelist()),
                             [f"example/{index:05d}.png" for index in range(3)])
            with Image.open(io.BytesIO(submission.read("example/00002.png"))) as mask:
                self.assertEqual(mask.size, (40, 32))
                self.assertEqual(mask.mode, "P")
        self.assertIsNone(_result_table({"mose_vos": result})[0]["mask_propagation"]["j_and_f"])
        args = SimpleNamespace(checkpoint=self.checkpoint, checkpoint_key="teacher", arch="auto",
                               datasets_root=self.root, seed=0, classification_manifests=None)
        self.assertIsNotNone(_load_completed_result(result_path, args, "mose_vos"))
        archive.unlink()
        self.assertIsNone(_load_completed_result(result_path, args, "mose_vos"))

    def test_archive_refuses_missing_predictions(self):
        predictions = self.root / "predictions"
        vos._save_mask(predictions / "example/00000.png", self.initial, self.palette)
        with self.assertRaisesRegex(FileNotFoundError, "Prediction missing"):
            vos._submission_archive(predictions, self.images.parent, ["example"], self.root / "submission.zip")

    def test_youtube_worker_retains_object_scores_for_seen_unseen_aggregation(self):
        dataset = self.root / "youtube_vos_2019/valid"
        images, masks = dataset / "JPEGImages/example", dataset / "Annotations/example"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        truth = self.initial.copy()
        truth[16:, 20:] = 11
        for index in range(3):
            Image.new("RGB", (40, 32), color="gray").save(images / f"{index:05d}.jpg")
            vos._save_mask(masks / f"{index:05d}.png", truth, self.palette)
        (dataset / "meta.json").write_text(json.dumps({"videos": {"example": {"objects": {
            "7": {"category": "dog", "frames": [f"{i:05d}" for i in range(3)]},
            "11": {"category": "novel", "frames": [f"{i:05d}" for i in range(3)]},
        }}}}))
        _, result = self.run_worker("youtube_vos")
        self.assertEqual(result["metrics_status"], "computed")
        self.assertEqual(set(result["videos"]["example"]["objects"]), {"7", "11"})
        self.assertGreater(result["metrics"]["j_and_f"], 0)


if __name__ == "__main__":
    unittest.main()
