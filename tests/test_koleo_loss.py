import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from losses import KoLeoLoss, iBOTLoss
from utils.checkpoint import _validate_resume_compatibility


def make_ibot_loss(koleo_regularizer):
    return iBOTLoss(
        out_dim=3, patch_out_dim=3, ngcrops=2, nlcrops=0,
        warmup_teacher_temp=0.07, teacher_temp=0.07,
        warmup_teacher_temp2=0.07, teacher_temp2=0.07,
        warmup_teacher_temp_epochs=0, nepochs=1,
        lambda3=0, koleo_regularizer=koleo_regularizer,
    )


class KoLeoLossTest(unittest.TestCase):
    def test_nearest_other_normalized_feature_and_gradient(self):
        features = torch.tensor(
            [[2.0, 0.0], [1.0, 1.0], [-1.0, 0.0]],
            requires_grad=True,
        )
        vectors = F.normalize(features, dim=-1)
        distances = torch.cdist(vectors, vectors)
        distances.fill_diagonal_(float("inf"))
        expected = -(distances.min(dim=1).values + 1e-8).log().mean()
        actual = KoLeoLoss()(features)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        actual.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_requires_two_samples_and_returns_float32_under_autocast(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            KoLeoLoss()(torch.randn(1, 3))
        features = torch.randn(3, 4, dtype=torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = KoLeoLoss()(features)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result))


class KoLeoIntegrationTest(unittest.TestCase):
    def test_two_crops_are_regularized_separately_at_original_weight(self):
        student = (torch.randn(4, 3, requires_grad=True),
                   torch.randn(4, 4, 3, requires_grad=True))
        teacher = (torch.randn(4, 3), torch.randn(4, 4, 3))
        masks = [torch.ones(2, 2, 2, dtype=torch.bool) for _ in range(2)]
        features = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0],
             [1.0, 0.0], [1.0, 1.0]],
            requires_grad=True,
        )
        loss = make_ibot_loss(True)
        targets = loss.softmax_center_teacher(teacher, 0.07, 0.07)
        result = loss(student, targets, None, masks, None,
                      student_cls_features=features)
        expected_raw = KoLeoLoss()(features[:2]) + KoLeoLoss()(features[2:])
        torch.testing.assert_close(result["koleo_raw"], expected_raw)
        torch.testing.assert_close(result["koleo"], expected_raw * 0.1)
        torch.testing.assert_close(
            result["loss"], result["cls"] + result["patch"] + result["koleo"]
        )
        self.assertEqual(result["koleo_active"].item(), 1)
        expected_distance = math.sqrt(2 - math.sqrt(2))
        self.assertAlmostEqual(
            result["koleo_raw"].item(), -2 * math.log(expected_distance), places=6
        )
        result["loss"].backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "pre-head student CLS"):
            loss(student, targets, None, masks, None)

        disabled = make_ibot_loss(False)
        control = disabled(student, targets, None, masks, None)
        self.assertEqual(control["koleo"].item(), 0)
        self.assertEqual(control["koleo_active"].item(), 0)
        torch.testing.assert_close(control["loss"], control["cls"] + control["patch"])

    def test_old_resume_defaults_to_disabled_but_cannot_enable_koleo(self):
        checkpoint = {"args": {"lambda3": 0}}
        _validate_resume_compatibility(
            checkpoint, SimpleNamespace(lambda3=0, koleo_regularizer=False)
        )
        with self.assertRaisesRegex(ValueError, "koleo_regularizer"):
            _validate_resume_compatibility(
                checkpoint, SimpleNamespace(lambda3=0, koleo_regularizer=True)
            )


if __name__ == "__main__":
    unittest.main()
