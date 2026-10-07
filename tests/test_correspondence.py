"""Regression checks for Probe3D matching and iBOT feature conventions."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from evaluation.utils.correspondence import (
    _spair_dataset_class, _spair_errors, _valid_flattened_features, binned_pair_recall,
    nearest_ratio_matches, patch_features,
)
from evaluation.utils.correspondence_features import CorrespondenceFeatures, fit_voc_standardizer, load_patch_projection


class CorrespondenceTest(unittest.TestCase):
    def test_standard_scaler_fits_only_selected_training_images(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import TensorDataset, Subset
        class Extractor:
            patch_size = 16
            def extract_tokens(self, images, apply_scaler=True):
                self.assert_no_scaler = not apply_scaler
                return images
        all_images = torch.tensor([[[0., 2.], [2., 4.]], [[1000., 1000.], [2000., 2000.]],
                                   [[4., 6.], [6., 8.]]])
        dataset = TensorDataset(all_images, torch.zeros(3))
        train = Subset(dataset, [0, 2])
        expected = StandardScaler().fit(all_images[[0, 2]].reshape(-1, 2).numpy())
        extractor = Extractor()
        with tempfile.TemporaryDirectory() as directory, \
                patch("evaluation.utils.dense._build_dense_datasets", return_value={"train": train}), \
                patch("evaluation.utils.datasets.segmentation_manifest", return_value={}):
            protocol = fit_voc_standardizer(extractor, Path("unused"), Path(directory), "cpu", num_workers=0)
            np.testing.assert_allclose(extractor.scaler_mean.numpy(), expected.mean_)
            np.testing.assert_allclose(extractor.scaler_scale.numpy(), expected.scale_)
            self.assertTrue(extractor.assert_no_scaler)
            self.assertEqual(protocol["fit_patch_vectors"], 4)
            self.assertFalse(protocol["correspondence_test_data_used_for_fit"])
            with np.load(Path(directory) / "standard_scaler.npz") as stats:
                np.testing.assert_array_equal(stats["image_indices"], [0, 2])

    def test_feature_variants_apply_norm_scaler_head_and_softmax_at_requested_points(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
                self.norm = torch.nn.LayerNorm(3)
            def prepare_tokens(self, images):
                return torch.arange(21, dtype=torch.float32).reshape(1, 7, 3)
        backbone = Backbone()
        image = torch.zeros(1, 3, 32, 32)
        normalized = backbone.norm(backbone.prepare_tokens(image))[:, 3:]
        extractor = CorrespondenceFeatures(backbone, "final_norm_standardized", 16, 2)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            extractor.extract_tokens(image)
        extractor.scaler_mean = torch.tensor([1., 2., 3.])
        extractor.scaler_scale = torch.tensor([2., 3., 4.])
        torch.testing.assert_close(extractor.extract_tokens(image), (normalized - extractor.scaler_mean) / extractor.scaler_scale)
        head = torch.nn.Linear(3, 5)
        projected = CorrespondenceFeatures(backbone, "projection", 16, 2, head)
        torch.testing.assert_close(projected.extract_tokens(image), head(normalized))
        softmax = CorrespondenceFeatures(backbone, "projection_softmax", 16, 2, head)
        torch.testing.assert_close(softmax.extract_tokens(image), head(normalized).softmax(-1))
        torch.testing.assert_close(softmax.extract_tokens(image).sum(-1), torch.ones(1, 4))

    def test_concatenation_uses_raw_blocks_4_6_8_12_in_order(self):
        class Increment(torch.nn.Module):
            def forward(self, x):
                return x + 1
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Increment() for _ in range(12)])
            def prepare_tokens(self, images):
                return torch.zeros(1, 5, 2)
            def norm(self, tokens):
                raise AssertionError("Raw block concatenation must not apply final LN")
        extractor = CorrespondenceFeatures(Backbone(), "concat_4_6_8_12", 16)
        actual = extractor.extract_tokens(torch.zeros(1, 3, 32, 32))
        torch.testing.assert_close(actual[0, 0], torch.tensor([4., 4., 6., 6., 8., 8., 12., 12.]))
        self.assertEqual(actual.shape, (1, 4, 8))

    def test_projection_loader_restores_shared_and_separate_trained_patch_paths(self):
        from model.head import iBOTHead
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory() as directory:
                original = iBOTHead(6, 7, patch_out_dim=7, hidden_dim=12, bottleneck_dim=4,
                                    shared_head=shared, norm_last_layer=False)
                original.eval()
                path = Path(directory) / "checkpoint.pth"
                torch.save({"teacher": {"module.head." + k: v for k, v in original.state_dict().items()}}, path)
                loaded, metadata = load_patch_projection(path, "teacher")
                vectors = torch.randn(9, 6)
                torch.testing.assert_close(loaded(vectors), original.project_patches(vectors))
                self.assertTrue(metadata["strict_checkpoint_load"])
                self.assertEqual(metadata["trunk"], "mlp." if shared else "patch_mlp.")
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

    def test_annotation_cache_preserves_released_pair_sampling(self):
        from evaluation.vendor.probe3d.evals.datasets.spair import SPairDataset
        annotations = [{"filename": str(i), "category": "cat", "viewpoint_variation": i % 3}
                       for i in range(30)]
        original_order = [row["filename"] for row in annotations]
        with patch.object(SPairDataset, "get_pair_annotations", return_value=annotations) as read_pairs, \
                patch.object(SPairDataset, "get_image_annotations", return_value={}) as read_images:
            cached_class = _spair_dataset_class()
            first = cached_class("unused", "test", class_name="cat", num_instances=7, vp_diff=0)
            second = cached_class("unused", "test", class_name="cat", num_instances=7, vp_diff=None)
            self.assertEqual(read_pairs.call_count, 1)
            self.assertEqual(read_images.call_count, 1)
            baseline_first = SPairDataset("unused", "test", class_name="cat", num_instances=7, vp_diff=0)
            baseline_second = SPairDataset("unused", "test", class_name="cat", num_instances=7, vp_diff=None)
            self.assertEqual(first.instances, baseline_first.instances)
            self.assertEqual(second.instances, baseline_second.instances)
            self.assertEqual([row["filename"] for row in annotations], original_order)

    def test_raw_block_features_exclude_registers_and_do_not_normalize(self):
        class Backbone:
            def prepare_tokens(self, images):
                return torch.arange(21, dtype=torch.float32).reshape(1, 7, 3)

            blocks = (lambda tokens: tokens * 2 + 1,)

            def get_intermediate_layers(self, *args, **kwargs):
                raise AssertionError("Probe3D must not apply final LayerNorm")

        actual = patch_features(Backbone(), torch.zeros(1, 3, 32, 32), 16, 2)
        expected = (torch.arange(21).reshape(1, 7, 3) * 2 + 1)[:, 3:]
        torch.testing.assert_close(actual, expected.float().transpose(1, 2).reshape(1, 3, 2, 2))

    def test_navi_interpolates_raw_features_before_cosine_normalization(self):
        features = torch.tensor([[[1., 100.], [3., 4.]], [[2., 1.], [7., 9.]]])
        grid = torch.ones(3, 5, 5)
        grid[2, 0, 0] = 0
        vectors, xyz = _valid_flattened_features(features, grid)
        expected = F.interpolate(features[None], size=(5, 5), mode="bicubic")[0]
        torch.testing.assert_close(vectors, expected.permute(1, 2, 0)[grid[2] > 0])
        self.assertEqual(xyz.shape, (24, 3))
        normalized_first = F.interpolate(F.normalize(features, dim=0)[None], size=(5, 5), mode="bicubic")[0]
        self.assertFalse(torch.allclose(vectors, normalized_first.permute(1, 2, 0)[grid[2] > 0]))

    def test_ratio_matches_equal_dense_cosine_distance_reference(self):
        torch.manual_seed(6)
        source, target = torch.randn(31, 12), torch.randn(37, 12)
        similarity = F.normalize(source, dim=1) @ F.normalize(target, dim=1).T
        values, ids = similarity.topk(2, dim=1)
        distances = (1 - values).clamp_min(1e-9)
        weights = 1 - distances[:, 0] / distances[:, 1]
        expected = weights.topk(11).indices
        for chunk in (1, 7, 1024):
            src, dst = nearest_ratio_matches(source, target, 11, chunk)
            torch.testing.assert_close(src, expected)
            torch.testing.assert_close(dst, ids[expected, 0])

    def test_spair_normalizes_each_token_and_filters_missing_keypoints(self):
        # Without per-token normalization, the high-norm diagonal distractor
        # wins over the more similar target at (0, 0).
        target = torch.tensor([[[2., 100.], [0., -1.]], [[1., 100.], [1., 0.]]])
        source = torch.zeros_like(target)
        source[0] = 1
        keypoints = torch.tensor([[0., 0., 1.], [800., 800., 0.]])
        errors = _spair_errors(torch.stack((source, target)), keypoints, keypoints, 1., 800)
        torch.testing.assert_close(errors, torch.zeros(1))

    def test_angle_bins_are_half_open_and_average_pairs(self):
        scores = binned_pair_recall([(0, .5), (29.9, 1), (30, .2), (60, .3), (90, .4)], (0, 30, 60, 90, 120))
        self.assertEqual(scores, {"0-30": 75., "30-60": 20., "60-90": 30., "90-120": 40.})


if __name__ == "__main__":
    unittest.main()
