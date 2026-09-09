import datetime
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from losses.region_loss import RegionLoss, intersection_patch_weights


def reference_overlap_targets(logits, boxes, min_area, temperature):
    """Dense NumPy-SK reference to verify the sparse gather/pool implementation."""
    weights, valid, _ = intersection_patch_weights(boxes, logits.shape[1], min_area)
    selected = weights.transpose(0, 1).reshape(logits.shape[:2]) > 0
    probabilities = torch.zeros_like(logits, dtype=torch.float32)
    if selected.any():
        matrix = np.exp(logits[selected].detach().float().numpy() / temperature).T
        prototypes, tokens = matrix.shape
        matrix /= matrix.sum()
        for _ in range(3):
            matrix /= matrix.sum(axis=1, keepdims=True) * prototypes
            matrix /= matrix.sum(axis=0, keepdims=True) * tokens
        probabilities[selected] = torch.from_numpy((matrix * tokens).T.copy())
    views = probabilities.chunk(2)
    pooled = torch.stack([
        (views[view] * weights[:, view, :, None]).sum(1)
        / weights[:, view].sum(-1, keepdim=True).clamp_min(1e-12)
        for view in range(2)
    ], dim=1)
    return pooled, weights, valid


def overlap_boxes():
    return torch.tensor([
        [[0., 0., 1., 1., 0.], [0.5, 0.25, 1., 1., 0.]],
        [[0., 0., 0.4, 1., 0.], [0.6, 0., 1., 1., 0.]],
    ])


def distributed_overlap_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        generator = torch.Generator().manual_seed(31)
        raw = torch.randn(2, 4, 4, 3, generator=generator) * 0.1
        region_loss = RegionLoss(0.1)
        # Test different valid counts, no local overlap, and no global overlap.
        for valid_samples in (3, 2, 0):
            boxes = overlap_boxes()[1:].repeat(4, 1, 1)
            boxes[:valid_samples] = overlap_boxes()[0]
            # Give different valid overlaps different numbers of selected patches.
            if valid_samples > 2:
                boxes[2, 1] = boxes[2, 0]
            expected, _, _ = reference_overlap_targets(
                raw.flatten(0, 1), boxes, 0.1, 0.07
            )
            local_slice = slice(2 * rank, 2 * (rank + 1))
            local_raw = raw[:, local_slice].flatten(0, 1)
            local_boxes = boxes[local_slice]
            targets = region_loss.sinkhorn_knopp_teacher(local_raw, local_boxes, 0.07)
            torch.testing.assert_close(targets.probabilities, expected[local_slice])

            # All ranks must finish region loss and backward, including empty ranks.
            student = tuple(torch.zeros(2, 4, 3, requires_grad=True) for _ in range(2))
            centered = tuple(torch.full((2, 4, 3), 1 / 3) for _ in range(2))
            result = region_loss(
                student, centered, local_boxes, teacher_overlap_targets=targets
            )
            result["loss"].backward()
            if targets.valid.any():
                expected_loss = np.log(3) * 2 * int(targets.valid.sum()) / valid_samples
                torch.testing.assert_close(
                    result["loss"], result["loss"].new_tensor(expected_loss)
                )
            else:
                assert result["loss"].item() == 0
                assert all(logits.grad.count_nonzero() == 0 for logits in student)
    finally:
        dist.destroy_process_group()


class IntersectionPatchWeightsTest(unittest.TestCase):
    def test_full_overlap_covers_every_patch_equally(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, area = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(weights, torch.full((1, 2, 4), 0.25))
        self.assertTrue(valid.item())
        torch.testing.assert_close(area, torch.tensor([1.0]))

    def test_intersection_is_projected_into_each_crop(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, area = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(
            weights[:, 0], torch.tensor([[0.0, 0.25, 0.0, 0.25]])
        )
        torch.testing.assert_close(weights[:, 1], torch.full((1, 4), 0.25))
        self.assertTrue(valid.item())
        torch.testing.assert_close(area, torch.tensor([0.5]))

    def test_horizontal_flip_mirrors_patch_coverage(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, _, _ = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(
            weights[:, 0], torch.tensor([[0.25, 0.0, 0.25, 0.0]])
        )

    def test_boundary_patches_receive_fractional_coverage(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.25, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, _, _ = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(
            weights[:, 0], torch.tensor([[0.125, 0.25, 0.125, 0.25]])
        )

    def test_overlap_below_threshold_is_skipped(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, _ = intersection_patch_weights(boxes, 4, 0.51)

        self.assertFalse(valid.item())
        self.assertEqual(weights.count_nonzero().item(), 0)

    def test_overlap_at_threshold_is_kept(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        _, valid, _ = intersection_patch_weights(boxes, 4, 0.5)

        self.assertTrue(valid.item())

    def test_zero_area_is_skipped_even_with_zero_threshold(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 0.4, 1.0, 0.0], [0.6, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, _ = intersection_patch_weights(boxes, 4, 0.0)

        self.assertFalse(valid.item())
        self.assertEqual(weights.count_nonzero().item(), 0)


class RegionLossTest(unittest.TestCase):
    def test_loss_matches_region_across_opposite_views(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]]]
        )
        teacher = (
            torch.tensor([[[1.0, 0.0]]]).expand(1, 4, 2),
            torch.tensor([[[0.0, 1.0]]]).expand(1, 4, 2),
        )
        matching_student = (
            torch.tensor([[[-10.0, 10.0]]]).expand(1, 4, 2),
            torch.tensor([[[10.0, -10.0]]]).expand(1, 4, 2),
        )
        nonmatching_student = tuple(-logits for logits in matching_student)
        loss_function = RegionLoss(min_area=0.05)

        matching = loss_function(matching_student, teacher, boxes)["loss"]
        nonmatching = loss_function(nonmatching_student, teacher, boxes)["loss"]

        self.assertLess(matching.item(), 1e-6)
        self.assertGreater(nonmatching.item(), 10.0)

    def test_invalid_batch_has_graph_connected_zero_loss(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 0.4, 1.0, 0.0], [0.6, 0.0, 1.0, 1.0, 0.0]]]
        )
        student = tuple(
            torch.randn(1, 4, 3, requires_grad=True) for _ in range(2)
        )
        teacher = tuple(
            torch.softmax(torch.randn(1, 4, 3), dim=-1)
            for _ in range(2)
        )

        result = RegionLoss(min_area=0.0)(student, teacher, boxes)
        result["loss"].backward()

        self.assertEqual(result["loss"].item(), 0.0)
        self.assertEqual(result["valid_ratio"].item(), 0.0)
        for logits in student:
            self.assertEqual(logits.grad.count_nonzero().item(), 0)


class OverlapSinkhornTest(unittest.TestCase):
    def test_joint_raw_patch_sk_then_geometric_pooling_matches_reference(self):
        logits = torch.randn(4, 4, 3, generator=torch.Generator().manual_seed(19)) * 0.1
        logits.requires_grad_()
        before = logits.detach().clone()
        boxes = overlap_boxes()

        actual = RegionLoss(0.1).sinkhorn_knopp_teacher(logits, boxes, 0.07)
        expected, weights, valid = reference_overlap_targets(logits, boxes, 0.1, 0.07)

        torch.testing.assert_close(actual.probabilities, expected)
        torch.testing.assert_close(actual.patch_weights, weights)
        torch.testing.assert_close(actual.valid, valid)
        torch.testing.assert_close(
            actual.probabilities.sum(-1), valid[:, None].expand(-1, 2).float()
        )
        torch.testing.assert_close(logits, before)
        self.assertFalse(actual.probabilities.requires_grad)

    def test_nonoverlap_and_invalid_sample_logits_cannot_affect_sk(self):
        logits = torch.randn(4, 4, 3) * 0.1
        boxes = overlap_boxes()
        region_loss = RegionLoss(0.1)
        before = region_loss.sinkhorn_knopp_teacher(logits, boxes, 0.07)
        excluded = before.patch_weights.transpose(0, 1).reshape(logits.shape[:2]) == 0
        changed = logits.clone()
        changed[excluded] = torch.tensor([10000., -10000., 5000.])

        after = region_loss.sinkhorn_knopp_teacher(changed, boxes, 0.07)

        torch.testing.assert_close(after.probabilities, before.probabilities)
        self.assertTrue(torch.isfinite(after.probabilities).all())

    def test_overlap_ce_uses_sk_targets_and_preserves_student_pooling(self):
        generator = torch.Generator().manual_seed(5)
        raw = torch.randn(4, 4, 3, generator=generator) * 0.1
        student = tuple(
            torch.randn(2, 4, 3, generator=generator, requires_grad=True)
            for _ in range(2)
        )
        boxes = overlap_boxes()
        region_loss = RegionLoss(0.1)
        sk = region_loss.sinkhorn_knopp_teacher(raw, boxes, 0.07)
        # Deliberately inconsistent centered probabilities must not enter SK CE.
        centered = tuple(torch.full((2, 4, 3), 1 / 3) for _ in range(2))

        result = region_loss(student, centered, boxes, teacher_overlap_targets=sk)

        expected_teacher, weights, _ = reference_overlap_targets(raw, boxes, 0.1, 0.07)
        expected_student = [
            (logits[0].softmax(-1) * weights[0, view, :, None]).sum(0)
            / weights[0, view].sum()
            for view, logits in enumerate(student)
        ]
        expected = -0.5 * (
            (expected_teacher[0, 0] * expected_student[1].log()).sum()
            + (expected_teacher[0, 1] * expected_student[0].log()).sum()
        )
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        for logits in student:
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(logits.grad[1].count_nonzero().item(), 0)

    def test_no_valid_overlap_returns_zero_targets_and_graph_connected_zero(self):
        boxes = overlap_boxes()[1:]
        student = tuple(torch.randn(1, 4, 3, requires_grad=True) for _ in range(2))
        raw = torch.randn(2, 4, 3)
        centered = tuple(logits.softmax(-1) for logits in raw.chunk(2))
        region_loss = RegionLoss(0.1)
        targets = region_loss.sinkhorn_knopp_teacher(raw, boxes, 0.07)

        result = region_loss(student, centered, boxes, teacher_overlap_targets=targets)
        result["loss"].backward()

        self.assertEqual(targets.probabilities.count_nonzero().item(), 0)
        self.assertEqual(result["loss"].item(), 0)
        for logits in student:
            self.assertEqual(logits.grad.count_nonzero().item(), 0)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_distributed_overlap_matches_joint_batch_including_empty_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                distributed_overlap_worker,
                args=((Path(directory) / "rendezvous").as_uri(),),
                nprocs=2, join=True,
            )


if __name__ == "__main__":
    unittest.main()
