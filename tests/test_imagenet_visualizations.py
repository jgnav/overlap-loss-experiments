import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

import imagenet_visualizations as viz


class ImageNetVisualizationTest(unittest.TestCase):
    def test_augmented_views_share_a_real_geometric_overlap(self):
        with mock.patch.object(viz, "VIS_RESOLUTION", 100):
            images, boxes = viz.make_views(Image.new("RGB", (100, 100)))
        self.assertEqual(set(images), set(viz.VIEW_NAMES))
        self.assertEqual(boxes["view_1"], (0, 0, 80, 80))
        self.assertEqual(boxes["view_2"], (20, 20, 100, 100))
        self.assertEqual(images["view_1"].size, images["view_2"].size)

    def test_cls_and_patch_features_are_separated_after_layer_average(self):
        class Backbone:
            def get_num_layers(self):
                return 4

            def get_intermediate_layers(self, tensor, n):
                self.requested_layers = n
                base = torch.tensor([[[10., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.]]])
                return [base + shift for shift in (0, 2, 4, 6)]

        model = Backbone()
        with (
            mock.patch.object(viz, "VIS_RESOLUTION", 4),
            mock.patch.object(viz, "DEVICE", torch.device("cpu")),
        ):
            tokens = viz.extract_tokens(model, Image.new("RGB", (4, 4)), patch_size=2)
        self.assertEqual(model.requested_layers, 4)
        np.testing.assert_allclose(tokens.cls, [13, 3])
        self.assertEqual(tokens.patches.shape, (4, 2))
        np.testing.assert_allclose(tokens.patches[0], [4, 3])
        np.testing.assert_allclose(tokens.output_cls, [16, 6])
        np.testing.assert_allclose(tokens.output_patches[0], [7, 6])

    def test_similarity_heatmap_contains_only_green_scores(self):
        axis = mock.Mock()
        scores = np.array([-1., -.5, .5, 1.])
        with mock.patch.object(viz, "VIS_RESOLUTION", 4):
            mappable = viz._heatmap_axis(axis, scores, 2, "cosine")
        self.assertIs(mappable, axis.imshow.return_value)
        axis.imshow.assert_called_once()
        np.testing.assert_array_equal(axis.imshow.call_args.args[0], scores.reshape(2, 2))
        options = axis.imshow.call_args.kwargs
        self.assertEqual(options["cmap"], "Greens")
        self.assertEqual((options["vmin"], options["vmax"]), (-1, 1))
        self.assertNotIn("alpha", options)

    def test_joint_pca_alignment_uses_original_to_color_all_views(self):
        rng = np.random.default_rng(4)
        reference = {name: rng.normal(size=(16, 3)) for name in viz.VIEW_NAMES}
        target = {
            name: scores[:, [2, 0, 1]] * np.array([-1, 1, -1])
            for name, scores in reference.items()
        }
        aligned, record = viz.align_pca_triplet(reference, target)
        for name in viz.VIEW_NAMES:
            np.testing.assert_allclose(aligned[name], reference[name])
        self.assertEqual(len(record["component_permutation"]), 3)

    def test_pca_uses_l2_normalized_unwhitened_patch_scores(self):
        base = np.array([[1., 2., 1.], [2., 1., 1.], [1., 1., 3.], [3., 1., 2.]])
        patches = {
            "original": base,
            "view_1": base[::-1] * np.arange(2., 6.)[:, None],
            "view_2": np.roll(base, 1, axis=0) * np.arange(3., 7.)[:, None],
        }
        tokens = {
            name: viz.Tokens(np.full(3, 1e6), values, np.full(3, 1e6), values)
            for name, values in patches.items()
        }
        projected = viz.pca_triplet(tokens)
        scores = np.concatenate([projected[name] for name in viz.VIEW_NAMES])
        features = np.concatenate([patches[name] for name in viz.VIEW_NAMES])
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        centered = normalized - normalized.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(scores.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(scores @ scores.T, centered @ centered.T, atol=1e-6)

    def test_pca_rgb_uses_one_scale_for_models_views_and_channels(self):
        scores = np.array([[4., .4, .04], [-4., -.4, -.04],
                           [2., .2, .02], [-2., -.2, -.02]])
        maps = {
            "iBOT": {view: scores for view in viz.VIEW_NAMES},
            "Ours": {view: 2 * scores for view in viz.VIEW_NAMES},
        }
        with (
            mock.patch.object(viz, "PCA_COLOR_PERCENTILE", 100),
            mock.patch.object(viz, "VIS_RESOLUTION", 2),
        ):
            scale = viz.pca_color_scale(maps)
            rgb = np.asarray(viz.pca_to_rgb(scores, grid=2, scale=scale))
        self.assertEqual(scale, 8)
        channel_ranges = np.ptp(rgb.reshape(-1, 3).astype(int), axis=0)
        self.assertGreater(channel_ranges[0], channel_ranges[1])
        self.assertGreater(channel_ranges[1], channel_ranges[2])

    def test_correspondences_use_overlap_and_shared_region_labels(self):
        grid = 4
        features = np.eye(grid * grid)
        regions = np.arange(grid * grid) % 3
        matches = viz.top_correspondences(
            features, features,
            (0, 0, 100, 100), (0, 0, 100, 100),
            regions, regions, grid, top_k=5,
        )
        self.assertEqual(len(matches), 5)
        self.assertTrue(all(match["source_patch"] == match["target_patch"] for match in matches))
        self.assertTrue(all(match["same_region"] for match in matches))


if __name__ == "__main__":
    unittest.main()
