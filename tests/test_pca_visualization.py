import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

import pca_visualization as visualization
from model.vision_transformer import VisionTransformer


class DenseFeatureAveragingTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 32))
        patcher = mock.patch.object(visualization, "DEVICE", torch.device("cpu"))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _model(layers):
        model = mock.Mock()
        model.get_num_layers.return_value = 12
        model.get_intermediate_layers.return_value = layers
        return model

    def test_averages_four_layers_in_fp32_and_excludes_cls(self):
        base = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        layers = [(base + offset).to(torch.bfloat16) for offset in (2, 4, 8, 16)]
        for layer in layers:
            layer[:, 0] = -1000
        model = self._model(layers)

        with mock.patch.object(visualization, "N_LAST_LAYERS", 4):
            features = visualization._extract_dense_features(model, self.image)

        np.testing.assert_array_equal(features, (base[0, 1:] + 7.5).numpy())
        self.assertEqual(features.shape, (4, 3))
        self.assertEqual(features.dtype, np.float32)
        model.get_intermediate_layers.assert_called_once()
        self.assertEqual(model.get_intermediate_layers.call_args.kwargs, {"n": 4})

    def test_one_layer_preserves_previous_features(self):
        layer = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        model = self._model([layer])

        with mock.patch.object(visualization, "N_LAST_LAYERS", 1):
            features = visualization._extract_dense_features(model, self.image)

        np.testing.assert_array_equal(features, layer[0, 1:].numpy())
        self.assertEqual(model.get_intermediate_layers.call_args.kwargs, {"n": 1})

    def test_invalid_layer_counts_are_rejected_before_forward(self):
        model = self._model([])
        for count in (0, -1, 13, 1.5, True):
            with self.subTest(count=count):
                with mock.patch.object(visualization, "N_LAST_LAYERS", count):
                    with self.assertRaisesRegex(ValueError, "N_LAST_LAYERS"):
                        visualization._extract_dense_features(model, self.image)
        model.get_intermediate_layers.assert_not_called()

    def test_non_square_patch_grid_is_rejected(self):
        model = self._model([torch.zeros(1, 4, 3)])
        with mock.patch.object(visualization, "N_LAST_LAYERS", 1):
            with self.assertRaisesRegex(ValueError, "Non-square patch-token grid"):
                visualization._extract_dense_features(model, self.image)

    def test_real_vit_matches_mean_of_normalized_block_outputs(self):
        model = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=12, depth=4, num_heads=3
        ).eval()
        tensor = visualization.MODEL_TRANSFORM(self.image).unsqueeze(0)
        with torch.inference_mode():
            layers = model.get_intermediate_layers(tensor, n=4)
            expected = torch.stack([layer[:, 1:] for layer in layers]).mean(0)

        with mock.patch.object(visualization, "N_LAST_LAYERS", 4):
            features = visualization._extract_dense_features(model, self.image)

        np.testing.assert_allclose(features, expected[0].numpy(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
