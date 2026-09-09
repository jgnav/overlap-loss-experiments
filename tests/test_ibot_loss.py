import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from losses import iBOTLoss
from losses.sinkhorn import sinkhorn_knopp


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


def reference_sinkhorn(logits, temperature):
    """Independent NumPy reference for DINOv2's three assignment iterations."""
    weights = np.exp(logits.detach().float().numpy() / temperature).T
    prototypes, tokens = weights.shape
    weights /= weights.sum()
    for _ in range(3):
        weights /= weights.sum(axis=1, keepdims=True) * prototypes
        weights /= weights.sum(axis=0, keepdims=True) * tokens
    return torch.from_numpy((weights * tokens).T.copy())


def distributed_sinkhorn_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(8, 5, generator=generator) * 0.1
        expected = reference_sinkhorn(logits, 0.09)
        # Exercise unequal token counts and a rank with no local tokens.
        for split in (1, 0):
            selection = slice(0, split) if rank == 0 else slice(split, 8)
            actual = sinkhorn_knopp(logits[selection], 0.09)
            torch.testing.assert_close(actual, expected[selection])
    finally:
        dist.destroy_process_group()


class AdaptationScheduleTest(unittest.TestCase):
    def test_teacher_temperatures_remain_at_pretrained_value(self):
        loss = make_loss()

        np.testing.assert_allclose(loss.teacher_temp_schedule, 0.07)
        np.testing.assert_allclose(loss.teacher_temp2_schedule, 0.07)


class TeacherNormalizationTest(unittest.TestCase):
    def test_center_softmax_and_ema_preserve_baseline_formula(self):
        loss = make_loss(patch_out_dim=5, center_momentum2=0.8)
        loss.center.copy_(torch.tensor([[0.1, -0.2, 0.3]]))
        loss.center2.copy_(torch.linspace(-0.1, 0.1, 5).reshape(1, 1, 5))
        cls = torch.randn(4, 3, requires_grad=True)
        patch = torch.randn(4, 4, 5, requires_grad=True)
        old_cls_center = loss.center.clone()
        old_patch_center = loss.center2.clone()

        targets = loss.softmax_center_teacher((cls, patch), 0.07, 0.09)

        torch.testing.assert_close(
            targets[0], ((cls - old_cls_center) / 0.07).softmax(-1)
        )
        torch.testing.assert_close(
            targets[1], ((patch - old_patch_center) / 0.09).softmax(-1)
        )
        self.assertFalse(targets[0].requires_grad)
        self.assertFalse(targets[1].requires_grad)
        torch.testing.assert_close(loss.center, old_cls_center)
        torch.testing.assert_close(loss.center2, old_patch_center)

        loss.update_center(cls, patch)

        torch.testing.assert_close(
            loss.center, old_cls_center * 0.9 + cls.mean(0) * 0.1
        )
        torch.testing.assert_close(
            loss.center2, old_patch_center * 0.8 + patch.mean((0, 1)) * 0.2
        )

    def test_sinkhorn_matches_reference_for_raw_selected_logits_in_fp32(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = (torch.randn(12, 5) * 0.1).to(dtype).requires_grad_()
                original = logits.detach().clone()

                actual = sinkhorn_knopp(logits, 0.09)

                torch.testing.assert_close(actual, reference_sinkhorn(logits, 0.09))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(
                    actual.sum(-1), torch.ones(actual.shape[:-1])
                )
                torch.testing.assert_close(logits, original)

    def test_sinkhorn_balances_constant_prototype_bias(self):
        logits = torch.tensor([[0.1, 0.2, 0.3]]).expand(4, -1)
        targets = sinkhorn_knopp(logits, 0.07)
        torch.testing.assert_close(targets, torch.full_like(targets, 1 / 3))

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_distributed_sinkhorn_matches_global_batch_with_unequal_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(
                distributed_sinkhorn_worker,
                args=(rendezvous,), nprocs=2, join=True,
            )


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
        targets = loss.softmax_center_teacher(teacher, 0.07, 0.07)

        result = loss(student, targets, None, masks, None)

        loss.region_loss.forward.assert_not_called()
        loss.update_center.assert_not_called()
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
        targets = loss.softmax_center_teacher(teacher, 0.07, 0.07)
        with mock.patch.object(
            loss.region_loss,
            "forward",
            wraps=loss.region_loss.forward,
        ) as region_forward:
            result = loss(student, targets, None, masks, crop_boxes)

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

    def test_both_modes_keep_centered_cls_and_patch_objectives(self):
        for mode in ("centering", "sinkhorn_knopp"):
            with self.subTest(mode=mode):
                loss = make_loss(nlcrops=1)
                student, teacher, masks, crop_boxes = self._inputs()
                local_cls = torch.randn(2, 3, requires_grad=True)
                # Include an unmasked patch and a fully unmasked sample.
                masks[0][0, 0, 0] = False
                masks[1][1] = False
                targets = loss.softmax_center_teacher(teacher, 0.07, 0.07)
                overlap_targets = (
                    loss.region_loss.sinkhorn_knopp_teacher(
                        teacher[1], crop_boxes, 0.07
                    )
                    if mode == "sinkhorn_knopp"
                    else None
                )
                teacher_cls = targets[0].chunk(2)
                teacher_patch = targets[1].chunk(2)
                student_cls = torch.cat((student[0], local_cls)).chunk(3)
                student_patch = student[1].chunk(2)
                expected_cls = torch.stack(
                    [
                        -(teacher_cls[q] * F.log_softmax(
                            student_cls[v] / 0.1, -1
                        )).sum(-1).mean()
                        for q in range(2)
                        for v in range(3)
                        if q != v
                    ]
                ).mean()
                expected_patch = []
                for q in range(2):
                    ce = -(teacher_patch[q] * F.log_softmax(
                        student_patch[q] / 0.1, -1
                    )).sum(-1)
                    mask = masks[q].flatten(1)
                    expected_patch.append(
                        ((ce * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
                    )

                with mock.patch.object(
                    loss.region_loss, "forward", wraps=loss.region_loss.forward
                ) as region:
                    result = loss(
                        student, targets, local_cls, masks, crop_boxes,
                        teacher_overlap_targets=overlap_targets,
                    )

                torch.testing.assert_close(result["cls"], expected_cls)
                torch.testing.assert_close(
                    result["patch"], torch.stack(expected_patch).mean()
                )
                for actual, expected in zip(region.call_args.args[1], teacher_patch):
                    torch.testing.assert_close(actual, expected)
                self.assertIs(
                    region.call_args.kwargs["teacher_overlap_targets"], overlap_targets
                )
                result["loss"].backward()
                for logits in (*student, local_cls):
                    self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
