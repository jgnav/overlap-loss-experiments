"""CPU checks for the newly added main-result evaluators."""

import unittest
import tempfile
import json
from types import SimpleNamespace
from pathlib import Path
from PIL import Image

import numpy as np
import torch

from evaluation.utils import correspondence as corr
from evaluation.utils import video_segmentation as vos
from evaluation.utils.classification_data import MULTILABEL_DATASETS, read_multilabel_manifest
from evaluation.prepare_visual_genome_manifest import build_manifest
from evaluation.utils.config import load_config
from evaluation.utils.orchestrator import EVALUATIONS, _result_table
from model.vision_transformer import VisionTransformer


ROOT = Path(__file__).resolve().parents[1]


class MainResultsEvaluationTest(unittest.TestCase):
    def test_public_vg500_lists_make_sparse_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "visual_genome/VG_100K"
            images.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                Image.new("RGB", (8, 8)).save(images / name)
            annotations = root / "public"
            annotations.mkdir()
            (annotations / "train_list_500.txt").write_text("a.jpg\n")
            (annotations / "test_list_500.txt").write_text("b.jpg\n")
            (annotations / "vg_category_500_labels_index.json").write_text(
                json.dumps({"a.jpg": [0, 2], "b.jpg": [1]}))
            manifest = build_manifest(root, annotations)
            self.assertEqual(len(manifest["classes"]), 500)
            self.assertEqual(manifest["splits"]["val"][0]["positive_indices"], [1])
            self.assertEqual(manifest["splits"]["train"][0]["image"],
                             "visual_genome/VG_100K/a.jpg")
            (annotations / "test_list_500.txt").write_text("a.jpg\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_manifest(root, annotations)

    def test_sparse_manifest_label_indices_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                Image.new("RGB", (8, 8)).save(root / name)
            path = root / "labels.json"
            manifest = {"dataset": "sample", "source": "synthetic",
                        "classes": ["a", "b"],
                        "splits": {"train": [
                            {"image": "a.jpg", "positive_indices": [0]},
                            {"image": "b.jpg", "positive_indices": [1]}],
                            "val": [{"image": "c.jpg", "positive_indices": [1]}]}}
            path.write_text(json.dumps(manifest))
            samples, _, _ = read_multilabel_manifest(path, root, "sample", 2)
            self.assertEqual(samples["train"][0][1], [1, 0])
            self.assertEqual(samples["train"][1][1], [0, 1])

    def test_default_suite_contains_every_table_column(self):
        config = load_config(ROOT / "config/evaluation.yaml")
        self.assertEqual(set(config.evaluations), {name for name, _, _ in EVALUATIONS})
        self.assertEqual(MULTILABEL_DATASETS["visual_genome"]["num_classes"], 500)
        self.assertEqual(MULTILABEL_DATASETS["visual_genome"]["epochs"], 200)
        self.assertEqual(len(EVALUATIONS), 22)

    def test_summary_has_correspondence_video_and_visual_genome_rows(self):
        results = {
            "visual_genome_multilabel": {"dataset": "Visual Genome VG500", "metrics": {"map_percent": 33.0}},
            "spair_correspondence": {"dataset": "SPair-71k", "task": "semantic_correspondence", "metrics": {"all": 30.0}},
            "davis_vos": {"dataset": "DAVIS 2017 val", "metrics": {"j_and_f": 61.0}},
        }
        table = _result_table(results)
        self.assertEqual([row["dataset"] for row in table],
                         ["Visual Genome VG500", "SPair-71k", "DAVIS 2017 val"])

    def test_dense_tokens_exclude_registers_once(self):
        backbone = VisionTransformer(img_size=[32], patch_size=16, embed_dim=12,
                                     depth=4, num_heads=3, num_register_tokens=2).eval()
        image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            dense = corr.patch_features(backbone, image, 16, 2)
            video, grid = vos._features(backbone, image[0], 16, 2)
        self.assertEqual(dense.shape, (1, 12, 2, 2))
        self.assertEqual(video.shape, (4, 12))
        self.assertEqual(grid, (2, 2))

    def test_ratio_matches_prefer_distinct_exact_features(self):
        source = torch.tensor([[1., 0.], [0., 1.]])
        target = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        source_index, target_index = corr.nearest_ratio_matches(source, target, 2, chunk_size=1)
        self.assertEqual(set(zip(source_index.tolist(), target_index.tolist())), {(0, 0), (1, 1)})

    def test_rotation_bins_are_half_open_pair_means(self):
        scored = [(0, 0.0), (20, 1.0), (30, 0.5), (60, 1.0), (90, 0.25)]
        self.assertEqual(corr.binned_pair_recall(scored, corr.NAVI_EDGES),
                         {"0-30": 50.0, "30-60": 50.0, "60-90": 100.0, "90-120": 25.0})

    def test_video_affinity_propagates_top1_patch_labels(self):
        features = torch.eye(4)
        masks = torch.tensor([[1., 0., 0., 0.], [0., 1., 1., 1.]])
        result = vos.propagate_labels(features, [features], [masks], (2, 2), radius=1, topk=1)
        torch.testing.assert_close(result[0].flatten(1), masks)

    def test_video_preflight_rejects_first_frame_only_validation_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "youtube_vos_2019/valid/JPEGImages/example"
            masks = root / "youtube_vos_2019/valid/Annotations/example"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            (images.parent.parent / "meta.json").write_text(json.dumps({"videos": {
                "example": {"objects": {"1": {"category": "dog", "frames": [
                    "00000", "00001", "00002"]}}}}}))
            for index in range(4):
                (images / f"{index:05d}.jpg").touch()
            (masks / "00000.png").touch()
            with self.assertRaisesRegex(FileNotFoundError, "scoring mask missing"):
                vos.preflight_masks(root, "youtube_vos")
            (masks / "00001.png").touch()
            with self.assertRaisesRegex(FileNotFoundError, "scoring mask missing"):
                vos.preflight_masks(root, "youtube_vos")
            (masks / "00002.png").touch()
            vos.preflight_masks(root, "youtube_vos")

    def test_youtube_metrics_balance_seen_and_unseen_objects(self):
        videos = {"a": {"objects": {"1": {"category": "dog"}, "2": {"category": "novel"}}}}
        per_video = {"a": {"objects": {"1": {"j": 0.8, "f": 0.6},
                                         "2": {"j": 0.2, "f": 0.4}}}}
        metrics = vos._youtube_metrics(per_video, videos)
        self.assertAlmostEqual(metrics["j_seen"], 80)
        self.assertAlmostEqual(metrics["f_unseen"], 40)
        self.assertAlmostEqual(metrics["j_and_f"], 50)

    def test_video_sequence_uses_initial_mask_and_scores_later_frames(self):
        class PositionBackbone:
            def get_intermediate_layers(self, images, n=4):
                tokens = torch.cat((torch.zeros(1, 1, 4), torch.eye(4)[None]), dim=1)
                return [tokens.to(images.device) for _ in range(n)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, masks = root / "frames", root / "masks"
            frames.mkdir(); masks.mkdir()
            truth = np.zeros((32, 32), dtype=np.uint8)
            truth[:16, :16] = 1
            for index in range(3):
                Image.new("RGB", (32, 32), color="gray").save(frames / f"{index:05d}.jpg")
                Image.fromarray(truth).save(masks / f"{index:05d}.png")
            scores, details = vos._score_video(
                PositionBackbone(), {"patch_size": 240, "num_register_tokens": 0},
                sorted(frames.iterdir()), masks, torch.device("cpu"), "davis",
            )
            self.assertEqual(details["scored_frames"], 1)
            self.assertGreater(scores["1"]["j"], 0.85)
            self.assertGreater(scores["1"]["f"], 0.7)

    def test_youtube_new_object_is_introduced_once(self):
        class PositionBackbone:
            def get_intermediate_layers(self, images, n=4):
                tokens = torch.cat((torch.zeros(1, 1, 4), torch.eye(4)[None]), dim=1)
                return [tokens.to(images.device) for _ in range(n)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, masks = root / "frames", root / "masks"
            frames.mkdir(); masks.mkdir()
            for index in range(3):
                Image.new("RGB", (32, 32), color="gray").save(frames / f"{index:05d}.jpg")
                truth = np.zeros((32, 32), dtype=np.uint8)
                truth[:16, :16] = 1
                if index:
                    truth[16:, 16:] = 2
                Image.fromarray(truth).save(masks / f"{index:05d}.png")
            scores, details = vos._score_video(
                PositionBackbone(), {"patch_size": 240, "num_register_tokens": 0},
                sorted(frames.iterdir()), masks, torch.device("cpu"), "youtube_vos",
            )
            self.assertEqual(details["objects"], [1, 2])
            self.assertIn("2", scores)
            self.assertGreater(scores["2"]["j"], 0.6)

    def test_davis_perfect_masks_score_one(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:40, 10:40] = 1
        j, f = vos._score_frame(mask, mask, (1,))[0]
        self.assertAlmostEqual(j, 1.0)
        self.assertAlmostEqual(f, 1.0)
