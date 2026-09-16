import datetime
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from losses.region_loss import RegionLoss, intersection_patch_fractions


def boxes_full(batch=1):
    return torch.tensor([[[0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.]]] * batch).reshape(batch, 2, 5)


def boxes_disjoint(batch=1):
    return torch.tensor([[[0., 0., .4, 1., 0.], [.6, 0., 1., 1., 0.]]] * batch)


def reference_ce(student, teacher, selected, temperature):
    # Deliberately use direct softmax -> arithmetic mean, independently of the
    # implementation's numerically stable log-space aggregation.
    s = [x[0, selected[v]].softmax(-1).mean(0) for v, x in
         enumerate([x / temperature for x in student])]
    t = [x[0, selected[v]].detach().softmax(-1).mean(0) for v, x in
         enumerate([x / temperature for x in teacher])]
    return -.5 * ((t[0] * s[1].log()).sum() + (t[1] * s[0].log()).sum())


def distributed_region_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=30))
    try:
        for valid_samples in (3, 2, 0):
            boxes = boxes_disjoint(4)
            boxes[:valid_samples] = boxes_full(valid_samples)
            local = slice(rank * 2, rank * 2 + 2)
            student = tuple(torch.zeros(2, 4, 3, requires_grad=True) for _ in range(2))
            teacher = tuple(torch.randn(2, 4, 3, requires_grad=True) for _ in range(2))
            result = RegionLoss()(student, teacher, boxes[local])
            result['loss'].backward()
            valid_local = max(0, min(2, valid_samples - rank * 2))
            expected = math.log(3) * 2 * valid_local / max(valid_samples, 1)
            torch.testing.assert_close(result['loss'], torch.tensor(expected))
            assert all(x.grad is None for x in teacher)
            for x in student:
                assert torch.isfinite(x.grad).all()
                assert x.grad[valid_local:].count_nonzero() == 0
            # DDP averages these rank-scaled losses/gradients.
            averaged = result['loss'].detach().clone()
            dist.all_reduce(averaged)
            torch.testing.assert_close(averaged / 2, torch.tensor(math.log(3) if valid_samples else 0.))
    finally:
        dist.destroy_process_group()


class IntersectionGeometryTest(unittest.TestCase):
    def test_full_overlap(self):
        fractions, valid, area = intersection_patch_fractions(boxes_full(), 4, .05)
        torch.testing.assert_close(fractions, torch.ones(1, 2, 4))
        self.assertTrue(valid.item())
        self.assertEqual(area.item(), 1.)

    def test_full_14_by_14_grid_at_strict_threshold(self):
        logits = tuple(torch.zeros(1, 196, 3) for _ in range(2))
        result = RegionLoss(patch_threshold=1.)(logits, logits, boxes_full())
        self.assertTrue(result['patch_mask'].all())

    def test_partial_patch_fractions_and_flip(self):
        boxes = boxes_full()
        boxes[0, 1, 0] = .25
        fractions, _, area = intersection_patch_fractions(boxes, 4, 0)
        torch.testing.assert_close(fractions[0, 0], torch.tensor([.5, 1., .5, 1.]))
        torch.testing.assert_close(fractions[0, 1], torch.ones(4))
        self.assertEqual(area.item(), .75)
        boxes[0, 0, 4] = 1
        flipped, _, _ = intersection_patch_fractions(boxes, 4, 0)
        torch.testing.assert_close(flipped[0, 0], torch.tensor([1., .5, 1., .5]))

    def test_area_filter_and_disjoint(self):
        boxes = boxes_full()
        boxes[0, 1, 0] = .5
        self.assertTrue(intersection_patch_fractions(boxes, 4, .5)[1].item())
        self.assertFalse(intersection_patch_fractions(boxes, 4, .51)[1].item())
        fractions, valid, _ = intersection_patch_fractions(boxes_disjoint(), 4, 0)
        self.assertFalse(valid.item())
        self.assertEqual(fractions.count_nonzero(), 0)


class RegionCompositionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.student = tuple((torch.randn(1, 4, 3) * .1).requires_grad_() for _ in range(2))
        self.teacher = tuple((torch.randn(1, 4, 3) * .1).requires_grad_() for _ in range(2))

    def test_equal_weight_mean_of_softmax_and_detached_teacher(self):
        boxes = boxes_full()
        boxes[0, 1, 0] = .25  # first view has 50% and 100% covered patches
        loss = RegionLoss(patch_threshold=.5, temperature=.2)
        result = loss(self.student, self.teacher, boxes)
        selected = torch.ones(2, 4, dtype=torch.bool)
        expected = reference_ce(self.student, self.teacher, selected, .2)
        torch.testing.assert_close(result['loss'], expected)
        reference_grad = torch.autograd.grad(expected, self.student)
        result['loss'].backward()
        for x, expected_grad in zip(self.student, reference_grad):
            torch.testing.assert_close(x.grad, expected_grad)
        self.assertTrue(all(x.grad is None for x in self.teacher))
        # Half-covered patches are equally weighted, not half-weighted.
        torch.testing.assert_close(result['patch_mask'], selected[None])

    def test_threshold_excludes_patches_and_their_gradients(self):
        boxes = boxes_full()
        boxes[0, 1, 0] = .25
        loss = RegionLoss(patch_threshold=.51, temperature=.2)
        selected = torch.tensor([[False, True, False, True], [True] * 4])
        result = loss(self.student, self.teacher, boxes)
        torch.testing.assert_close(result['patch_mask'], selected[None])
        torch.testing.assert_close(result['loss'], reference_ce(self.student, self.teacher, selected, .2))
        changed_s = tuple(x.detach().clone() for x in self.student)
        changed_t = tuple(x.detach().clone() for x in self.teacher)
        changed_s[0][:, ~selected[0]] = torch.tensor([1e6, -1e6, 0.])
        changed_t[0][:, ~selected[0]] = torch.tensor([-1e6, 1e6, 0.])
        torch.testing.assert_close(loss(changed_s, changed_t, boxes)['loss'], result['loss'])
        result['loss'].backward()
        self.assertEqual(self.student[0].grad[:, ~selected[0]].count_nonzero(), 0)

    def test_empty_selection_in_one_view_and_disjoint_are_skipped(self):
        narrow = boxes_full()
        narrow[0, 1, :4] = torch.tensor([.45, 0., .55, 1.])
        for boxes in (narrow, boxes_disjoint()):
            student = tuple(x.detach().clone().requires_grad_() for x in self.student)
            result = RegionLoss()(student, self.teacher, boxes)
            self.assertEqual(result['valid_ratio'].item(), 0)
            self.assertEqual(result['loss'].item(), 0)
            result['loss'].backward()
            self.assertTrue(all(x.grad.count_nonzero() == 0 for x in student))

    def test_repeating_patches_keeps_mean_and_swapping_views_keeps_loss(self):
        loss = RegionLoss(temperature=.2)
        original = loss(self.student, self.teacher, boxes_full())['loss']
        repeated_s = tuple(x.repeat_interleave(4, dim=1) for x in self.student)
        repeated_t = tuple(x.repeat_interleave(4, dim=1) for x in self.teacher)
        torch.testing.assert_close(loss(repeated_s, repeated_t, boxes_full())['loss'], original)
        torch.testing.assert_close(loss(self.student[::-1], self.teacher[::-1], boxes_full())['loss'], original)

    def test_extreme_logits_have_finite_loss_and_nonzero_gradients(self):
        student = tuple(torch.tensor([[[10000., -10000.]]], requires_grad=True) for _ in range(2))
        teacher = tuple(-x.detach() for x in student)
        loss = RegionLoss()(student, teacher, boxes_full())['loss']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for x in student:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_invalid_hyperparameters(self):
        for kwargs in ({'temperature': 0}, {'temperature': float('nan')},
                       {'patch_threshold': 0}, {'patch_threshold': 1.1}):
            with self.assertRaises(ValueError):
                RegionLoss(**kwargs)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo required')
    def test_distributed_empty_ranks_and_global_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_region_worker,
                     args=((Path(directory) / 'rendezvous').as_uri(),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
