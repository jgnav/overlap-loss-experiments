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
