import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

import patch_concept_visualization as viz


class PatchProbabilityTest(unittest.TestCase):
    def test_training_center_shape_preserves_patch_axis(self):
        logits = torch.tensor([[2., 0., 1.], [1., 3., 0.],
                               [0., 1., 3.], [3., 1., 0.]])
        center = torch.tensor([0.1, 0.2, 0.3])
        backbone = Mock(return_value=torch.zeros(1, 5, 2))
        backbone.num_register_tokens = 0
        backbone.patch_embed = SimpleNamespace(patch_size=16)
        model = viz.TeacherModel(
            backbone=backbone,
            head=Mock(return_value=(None, logits.unsqueeze(0))),
            metadata={},
            patch_temperature=0.7, patch_center=None, concepts=3,
        )
        labels = np.array([0, 1, 0, 1])
        expected = ((logits - center) / 0.7).softmax(-1)[[0, 2]].mean(0)
        with patch.object(viz, "DEVICE", torch.device("cpu")), \
             patch.object(viz, "KMeans") as kmeans, \
             patch.object(viz, "_select_object_cluster", return_value=0):
            kmeans.return_value.fit_predict.return_value = labels
            for shape in ((3,), (1, 3), (1, 1, 3)):
                with self.subTest(center_shape=shape):
                    model.patch_center = center.reshape(shape)
                    result = viz._extract_result(model, Image.new("RGB", (32, 32)))
                    np.testing.assert_allclose(result.probabilities, expected.numpy())
                    self.assertEqual(result.probabilities.shape, (3,))
                    np.testing.assert_array_equal(result.cluster_map, labels.reshape(2, 2))


if __name__ == "__main__":
    unittest.main()
