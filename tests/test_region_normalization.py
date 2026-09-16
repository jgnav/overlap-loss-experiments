import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from losses.region_loss import RegionLoss, intersection_patch_fractions
from losses.sinkhorn import sinkhorn_log_probabilities
from tests.test_region_loss import boxes_full, boxes_disjoint


def reference_sk(logits, temperature):
    q = (logits / temperature).exp()
    for _ in range(3):
        q = q / q.sum(dim=0, keepdim=True)
        q = q / q.sum(dim=1, keepdim=True)
    return q


def reference_loss(student, teacher, boxes, mode, temperature=.2):
    fractions, valid, _ = intersection_patch_fractions(boxes, 4, 0.)
    selected = fractions >= .51
    valid = valid & selected.any(-1).all(-1)
    selected = selected & valid[:, None, None]
    if not valid.any():
        return student.sum() * 0
    regions = []
    for x in (student, teacher.detach()):
        if mode == 'sinkhorn':
            normalized = torch.zeros_like(x)
            normalized[selected] = reference_sk(x[selected], temperature)
        elif mode == 'softmax':
            normalized = (x / temperature).softmax(-1)
        else:
            normalized = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        regions.append((normalized * selected[..., None]).sum(2)[valid]
                       / selected.sum(2)[valid, :, None])
    s, t = regions
    if mode == 'raw_logits':
        return (1 - .5 * (F.cosine_similarity(s[:, 1], t[:, 0])
                         + F.cosine_similarity(s[:, 0], t[:, 1]))).mean()
    return -.5 * ((t[:, 0] * s[:, 1].log()).sum(-1)
                  + (t[:, 1] * s[:, 0].log()).sum(-1)).mean()


def distributed_sk_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=30))
    try:
        for valid_count in (3, 2, 0):
            torch.manual_seed(37)
            student = (torch.randn(4, 2, 4, 3) * .1).requires_grad_()
            teacher = (torch.randn(4, 2, 4, 3) * .1).requires_grad_()
            boxes = boxes_disjoint(4)
            if valid_count:
                boxes[:valid_count] = boxes_full(valid_count)
                boxes[0, 1, 0] = .25  # unequal selected patch counts across ranks
            expected = reference_loss(student, teacher, boxes, 'sinkhorn')
            expected.backward()
            part = slice(2 * rank, 2 * rank + 2)
            local_student = student.detach()[part].clone().requires_grad_()
            local_teacher = teacher.detach()[part].clone().requires_grad_()
            loss = RegionLoss(patch_threshold=.51, temperature=.2, normalization='sinkhorn')
            result = loss(tuple(local_student.unbind(1)), tuple(local_teacher.unbind(1)), boxes[part])
            result['loss'].backward()
            # Each rank scales its local sum for DDP's eventual gradient average.
            torch.testing.assert_close(local_student.grad / 2, student.grad[part], atol=2e-6, rtol=2e-5)
            assert local_teacher.grad is None
            average = result['loss'].detach().clone()
            dist.all_reduce(average)
            torch.testing.assert_close(average / 2, expected.detach())
    finally:
        dist.destroy_process_group()


class RegionNormalizationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.student = (torch.randn(2, 2, 4, 3) * .1).requires_grad_()
        self.teacher = (torch.randn(2, 2, 4, 3) * .1).requires_grad_()
        self.boxes = boxes_full(2)
        self.boxes[0, 1, 0] = .25

    def test_all_modes_match_reference_loss_and_student_gradient(self):
        for mode in ('softmax', 'raw_logits', 'sinkhorn'):
            with self.subTest(mode=mode):
                student = self.student.detach().clone().requires_grad_()
                loss = RegionLoss(patch_threshold=.51, temperature=.2, normalization=mode)
                result = loss(tuple(student.unbind(1)), tuple(self.teacher.unbind(1)), self.boxes)
                expected = reference_loss(student, self.teacher, self.boxes, mode)
                expected_grad = torch.autograd.grad(expected, student)[0]
                result['loss'].backward()
                torch.testing.assert_close(result['loss'], expected)
                torch.testing.assert_close(student.grad, expected_grad, atol=2e-6, rtol=2e-5)
                self.assertIsNone(self.teacher.grad)
                self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_raw_vectors_invariant_to_positive_patch_scaling_and_temperature(self):
        loss = RegionLoss(normalization='raw_logits')
        original = loss(tuple(self.student.unbind(1)), tuple(self.teacher.unbind(1)), self.boxes)['loss']
        scale = torch.rand(2, 2, 4, 1) + 1
        loss.temperature = .0001
        actual = loss(tuple((self.student * scale).unbind(1)),
                      tuple((self.teacher * scale).unbind(1)), self.boxes)['loss']
        torch.testing.assert_close(actual, original)

    def test_sinkhorn_only_uses_selected_valid_patches(self):
        self.boxes[1] = boxes_disjoint()[0]
        loss = RegionLoss(patch_threshold=.51, normalization='sinkhorn')
        result = loss(tuple(self.student.unbind(1)), tuple(self.teacher.unbind(1)), self.boxes)
        selected = result['patch_mask']
        student = self.student.detach().clone()
        teacher = self.teacher.detach().clone()
        student[~selected] = torch.tensor([1e8, -1e8, 2.])
        teacher[~selected] = torch.tensor([-1e8, 1e8, 2.])
        actual = loss(tuple(student.unbind(1)), tuple(teacher.unbind(1)), self.boxes)['loss']
        torch.testing.assert_close(actual, result['loss'])
        result['loss'].backward()
        self.assertEqual(self.student.grad[~selected].count_nonzero(), 0)

    def test_sinkhorn_balances_prototype_bias_and_is_numerically_stable(self):
        logits = torch.tensor([[10000., -10000., 4000.]]).repeat(8, 1).requires_grad_()
        logs = sinkhorn_log_probabilities(logits, .1)
        torch.testing.assert_close(logs.exp(), torch.full_like(logits, 1 / 3))
        logs.square().mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_pairs_skip_sinkhorn_and_keep_zero_student_gradients(self):
        for mode in ('softmax', 'raw_logits', 'sinkhorn'):
            student = self.student.detach().clone().requires_grad_()
            loss = RegionLoss(normalization=mode)
            with mock.patch('losses.region_loss.sinkhorn_log_probabilities', side_effect=AssertionError):
                result = loss(tuple(student.unbind(1)), tuple(self.teacher.unbind(1)), boxes_disjoint(2))
            result['loss'].backward()
            self.assertEqual(result['loss'].item(), 0)
            self.assertEqual(student.grad.count_nonzero(), 0)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'region_normalization'):
            RegionLoss(normalization='centering')

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo required')
    def test_distributed_sinkhorn_gradients_match_global_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_sk_worker,
                     args=((Path(directory) / 'rendezvous').as_uri(),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
