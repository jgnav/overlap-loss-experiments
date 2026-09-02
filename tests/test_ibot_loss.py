import unittest
from unittest import mock

import numpy as np
import torch

from losses import iBOTLoss


def make_loss(**overrides):
    arguments = {
        "out_dim": 3,
        "patch_out_dim": 3,
        "ngcrops": 2,
        "nlcrops": 0,
        "warmup_teacher_temp": 0.07,
        "teacher_temp": 0.07,
        "warmup_teacher_temp2": 0.07,
        "teacher_temp2": 0.07,
        "warmup_teacher_temp_epochs": 0,
        "nepochs": 8,
        "lambda3": 0.1,
    }
    arguments.update(overrides)
    return iBOTLoss(**arguments)


class AdaptationScheduleTest(unittest.TestCase):
    def test_teacher_temperatures_remain_at_pretrained_value(self):
        loss = make_loss()

        np.testing.assert_allclose(loss.teacher_temp_schedule, 0.07)
        np.testing.assert_allclose(loss.teacher_temp2_schedule, 0.07)

class PureIBOTAndDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        batch_size = 2
        patch_count = 4
        dimension = 3
        student = (
            torch.randn(batch_size * 2, dimension, requires_grad=True),
            torch.randn(
                batch_size * 2,
                patch_count,
                dimension,
                requires_grad=True,
            ),
        )
        teacher = (
            torch.randn(batch_size * 2, dimension),
            torch.randn(batch_size * 2, patch_count, dimension),
        )
        masks = [
            torch.ones(batch_size, 2, 2, dtype=torch.bool),
            torch.ones(batch_size, 2, 2, dtype=torch.bool),
        ]
        crop_boxes = torch.tensor(
            [
                [[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0, 1.0, 0.0], [0.2, 0.0, 1.0, 1.0, 0.0]],
            ]
        )
        return student, teacher, masks, crop_boxes

    def test_lambda3_zero_never_calls_region_loss(self):
        loss = make_loss(lambda3=0.0)
        loss.update_center = mock.Mock()
        loss.region_loss.forward = mock.Mock(
            side_effect=AssertionError("RegionLoss must be inactive")
        )
        student, teacher, masks, crop_boxes = self._inputs()

        result = loss(student, teacher, None, masks, None, epoch=0)

        loss.region_loss.forward.assert_not_called()
        torch.testing.assert_close(result["loss"], result["cls"] + result["patch"])
        self.assertEqual(result["region_active"].item(), 0.0)
        self.assertEqual(result["region_raw"].item(), 0.0)
        self.assertEqual(result["region"].item(), 0.0)
        self.assertEqual(result["region_valid_ratio"].item(), 0.0)
        self.assertEqual(result["region_intersection_area"].item(), 0.0)

    def test_lambda3_positive_calls_region_loss_with_constant_weight(self):
        loss = make_loss(lambda3=0.2)
        loss.update_center = mock.Mock()
        student, teacher, masks, crop_boxes = self._inputs()
        with mock.patch.object(
            loss.region_loss,
            "forward",
            wraps=loss.region_loss.forward,
        ) as region_forward:
            result = loss(student, teacher, None, masks, crop_boxes, epoch=0)

        region_forward.assert_called_once()
        self.assertAlmostEqual(result["region_weight"].item(), 0.2)
        torch.testing.assert_close(
            result["region"], result["region_raw"] * 0.2
        )
        self.assertEqual(result["region_active"].item(), 1.0)
        self.assertGreater(result["region_raw"].item(), 0.0)
        for key in (
            "patch_masked",
            "patch_masked_inside_overlap",
            "patch_masked_outside_overlap",
            "student_patch_entropy",
            "teacher_patch_entropy",
            "student_patch_max_probability",
            "teacher_patch_max_probability",
            "student_patch_effective_prototypes",
            "teacher_patch_effective_prototypes",
        ):
            self.assertTrue(torch.isfinite(result[key]).item(), key)


if __name__ == "__main__":
    unittest.main()
