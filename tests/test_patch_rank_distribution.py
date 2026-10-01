import datetime
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from losses.patch_rank_distribution_loss import (
    PatchRankDistributionLoss, mean_patch_permutations, MAX_REFERENCE_REGIONS,
)
from losses.region_ordering_loss import collect_overlap_regions
from losses.region_sorting import bitonic_permutation
from tests.test_ibot_loss import make_loss
from train import load_config
from model.head import iBOTHead
from model.vision_transformer import VisionTransformer
from utils.training import MultiCropWrapper


def fixture(batch=4):
    torch.manual_seed(613)
    student = tuple(torch.randn(batch, 16, 6).requires_grad_() for _ in range(2))
    teacher = tuple(torch.randn(batch, 16, 6).requires_grad_() for _ in range(2))
    boxes = torch.tensor([[[0, 0, 1, 1, 0], [.25, 0, 1, 1, 1]]] * batch, dtype=torch.float32)
    return student, teacher, boxes


def ordered_references(owners, query, count, generator):
    return [row for owner, rows in owners.items() if owner != query for row in rows][:count]


def brute_force(student, teacher, boxes):
    regions, masks = collect_overlap_regions(boxes, [16, 16], global_only=True, min_area=.1)
    bank = []
    for image, entries in enumerate(regions):
        view = entries[0].teacher_view
        features = teacher[view][image, masks[view][image, 0]].detach()
        bank.append(F.normalize(F.normalize(features, dim=-1).mean(0), dim=0))
    bank = torch.stack(bank)
    losses = []
    for image in range(len(bank)):
        refs = bank[[other for other in range(len(bank)) if other != image]]
        matrices = []
        for views in (teacher, student):
            values = []
            for view in range(2):
                features = views[view][image, masks[view][image, 0]].float()
                if views is teacher:
                    features = features.detach()
                _, permutation = bitonic_permutation(F.normalize(features, dim=-1) @ refs.T)
                values.append(permutation.mean(0))
            matrices.append(values)
        for t, s in ((0, 1), (1, 0)):
            losses.append(-(matrices[0][t] * matrices[1][s].clamp_min(1e-12).log()).sum(0).mean())
    return torch.stack(losses).mean()


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        student, teacher, boxes = fixture(6)
        local_student = tuple(value.detach()[rank * 3:(rank + 1) * 3].requires_grad_() for value in student)
        local_teacher = tuple(value.detach()[rank * 3:(rank + 1) * 3] for value in teacher)
        with mock.patch('losses.patch_rank_distribution_loss.sample_external_references', ordered_references):
            result = PatchRankDistributionLoss()(local_student, local_teacher, boxes[rank * 3:(rank + 1) * 3])
        # Compare rank-local gradients after DDP's averaging with global reference.
        expected = brute_force(student, teacher, boxes)
        gradients = torch.autograd.grad(expected, student)
        result['loss'].backward()
        for actual, full in zip(local_student, gradients):
            torch.testing.assert_close(actual.grad / 2, full[rank * 3:(rank + 1) * 3], atol=2e-5, rtol=2e-4)
        assert result['query_count'].item() == 12
        boxes[:3, 1, :4] = torch.tensor([.9, .9, 1., 1.])
        result = PatchRankDistributionLoss()(local_student, local_teacher, boxes[rank * 3:(rank + 1) * 3])
        result['loss'].backward()
        assert result['query_count'].item() == 6
        if rank == 0:
            assert result['loss'].item() == 0
    finally:
        dist.destroy_process_group()


class PatchRankDistributionTest(unittest.TestCase):
    def test_tiled_matrix_means_and_gradients_match_direct_sorting(self):
        torch.manual_seed(59)
        similarities = torch.randn(269, 7, requires_grad=True)
        rows = torch.arange(269) % 3
        _, permutations = bitonic_permutation(similarities)
        direct = torch.stack([permutations[rows == index].mean(0) for index in range(3)])
        weights = torch.randn_like(direct)
        expected_gradient, = torch.autograd.grad((direct * weights).sum(), similarities)
        actual = mean_patch_permutations(similarities, rows, 3)
        (actual * weights).sum().backward()
        torch.testing.assert_close(actual, direct)
        torch.testing.assert_close(similarities.grad, expected_gradient, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual.sum(1), torch.ones(3, 7))
        torch.testing.assert_close(actual.sum(2), torch.ones(3, 7))

    def test_full_loss_and_gradient_with_flips_and_unequal_patch_counts(self):
        student, teacher, boxes = fixture()
        expected = brute_force(student, teacher, boxes)
        gradients = torch.autograd.grad(expected, student)
        with mock.patch('losses.patch_rank_distribution_loss.sample_external_references', ordered_references):
            result = PatchRankDistributionLoss()(student, teacher, boxes)
        torch.testing.assert_close(result['loss'], expected)
        result['loss'].backward()
        for value, gradient in zip(student, gradients):
            torch.testing.assert_close(value.grad, gradient, atol=2e-5, rtol=2e-4)
        self.assertTrue(all(value.grad is None for value in teacher))
        self.assertEqual(result['query_count'].item(), 8)
        self.assertEqual(result['references_per_query'].item(), 3)

    def test_equal_feature_means_can_have_different_ranking_distributions(self):
        patches = torch.tensor([[.95, .05], [.15, .85]])
        repeated_mean = patches.mean(0).expand(2, -1)
        bank = F.normalize(torch.tensor([[1., 0], [0., 1.], [-1., 0.]]), dim=-1)
        rows = torch.zeros(2, dtype=torch.long)
        a = mean_patch_permutations(F.normalize(patches, dim=-1) @ bank.T, rows, 1)
        b = mean_patch_permutations(F.normalize(repeated_mean, dim=-1) @ bank.T, rows, 1)
        torch.testing.assert_close(patches.mean(0), repeated_mean.mean(0))
        self.assertGreater((a - b).abs().max().item(), .3)

    def test_empty_overlap_insufficient_references_and_local_geometry_independence(self):
        student, teacher, boxes = fixture(2)
        result = PatchRankDistributionLoss()(student, teacher, boxes)
        self.assertEqual(result['loss'].item(), 0)
        result['loss'].backward()
        self.assertTrue(all(value.grad.count_nonzero() == 0 for value in student))
        student, teacher, boxes = fixture(4)
        extended = torch.cat([boxes, torch.rand(4, 10, 5)], 1)
        torch.testing.assert_close(PatchRankDistributionLoss()(student, teacher, boxes)['loss'],
                                   PatchRankDistributionLoss()(student, teacher, extended)['loss'])
        boxes[:, 1, :4] = torch.tensor([.9, .9, 1., 1.])
        result = PatchRankDistributionLoss()(student, teacher, boxes)
        self.assertEqual(result['query_count'].item(), 0)
        result['loss'].backward()
        self.assertTrue(all(value.grad.count_nonzero() == 0 for value in student))

    def test_cpu_autocast_seed_and_resume(self):
        student, teacher, boxes = fixture()
        loss = PatchRankDistributionLoss(seed=92)
        expected = loss(student, teacher, boxes)
        restored = PatchRankDistributionLoss(seed=92)
        restored.load_state_dict(loss.state_dict())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = PatchRankDistributionLoss(seed=92)(student, teacher, boxes)
        self.assertEqual(actual['loss'].dtype, torch.float32)
        torch.testing.assert_close(actual['loss'], expected['loss'], atol=0, rtol=0)
        torch.testing.assert_close(restored(student, teacher, boxes)['loss'], loss(student, teacher, boxes)['loss'])

    def test_config_and_baseline_objective_unchanged(self):
        base = Path('config/train.yaml').read_text()
        path = Path('config/ablations/loss_modality_patch_rank_distribution.yaml')
        self.assertEqual(path.read_text(), base.replace('loss_modality: standard',
                         'loss_modality: patch_rank_distribution').replace('batch_size_per_gpu: 64', 'batch_size_per_gpu: 48'))
        config = load_config(path)
        self.assertEqual((config.loss_modality, config.lambda3, config.additional_epochs), ('patch_rank_distribution', .4, 50))
        baseline = make_loss(lambda3=.4, region_normalization='centering', region_patch_threshold=1.)
        criterion = make_loss(lambda3=.4, region_normalization='centering', region_patch_threshold=1.,
                              loss_modality='patch_rank_distribution')
        criterion.load_state_dict(baseline.state_dict(), strict=False)
        sf, tf, boxes = fixture()
        outputs = (torch.randn(8, 3, requires_grad=True), torch.randn(8, 16, 3, requires_grad=True))
        teacher_logits = (torch.randn(8, 3), torch.randn(8, 16, 3))
        targets = baseline.softmax_center_teacher(teacher_logits, .07, .07)
        masks = [torch.ones(4, 4, 4, dtype=torch.bool)] * 2
        kwargs = dict(teacher_patch_logits=teacher_logits[1])
        ordinary = baseline(outputs, targets, None, masks, boxes, **kwargs)
        actual = criterion(outputs, targets, None, masks, boxes, student_patch_features=torch.cat(sf),
                           teacher_patch_features=torch.cat(tf), **kwargs)
        for key in ('cls', 'patch', 'region'):
            torch.testing.assert_close(actual[key], ordinary[key], atol=0, rtol=0)
        torch.testing.assert_close(actual['loss'], ordinary['loss'] + .1 * actual['region_ordering_raw'])
        self.assertFalse(criterion.needs_local_patch_logits)
        with self.assertRaisesRegex(ValueError, 'pre-head'):
            criterion(outputs, targets, None, masks, boxes, **kwargs)
        criterion.lambda3 = 0
        with mock.patch.object(criterion.ordering_loss, 'forward', side_effect=AssertionError('skip')):
            criterion(outputs, targets, None, masks, boxes, **kwargs)

    def test_backbone_receives_gradient_without_new_head_or_teacher_passes(self):
        def network(masked):
            return MultiCropWrapper(VisionTransformer(img_size=[32], patch_size=8, embed_dim=12,
                depth=1, num_heads=3, return_all_tokens=True, masked_im_modeling=masked, num_register_tokens=2),
                iBOTHead(12, 3, patch_out_dim=3, hidden_dim=16, bottleneck_dim=4, nlayers=2, norm_last_layer=False))
        student, teacher = network(True), network(False)
        teacher.load_state_dict(student.state_dict(), strict=False)
        images = [torch.randn(4, 3, 32, 32) for _ in range(2)]
        masks = [torch.zeros(4, 4, 4, dtype=torch.bool) for _ in range(2)]
        for mask in masks:
            mask[:, ::2] = True
        with torch.no_grad():
            teacher_features, teacher_output = teacher(images, return_backbone_feat=True)
        student_features, student_output = student(images, mask=masks, return_backbone_feat=True)
        criterion = make_loss(lambda3=.4, region_normalization='centering', region_patch_threshold=1.,
                              loss_modality='patch_rank_distribution')
        result = criterion(student_output, criterion.softmax_center_teacher(teacher_output, .07, .07),
            None, masks, fixture()[2], teacher_patch_logits=teacher_output[1],
            student_patch_features=student_features[:, 3:], teacher_patch_features=teacher_features[:, 3:])
        result['region_ordering'].backward()
        self.assertGreater(student.backbone.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in student.head.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertEqual(list(criterion.ordering_loss.parameters()), [])

    def test_teacher_sorting_not_repeated_and_no_patch_matrices_saved(self):
        student, teacher, boxes = fixture()
        stored = []
        def save(tensor):
            stored.append(tuple(tensor.shape))
            return tensor
        with mock.patch('losses.patch_rank_distribution_loss.bitonic_permutation', wraps=bitonic_permutation) as sorter:
            with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
                result = PatchRankDistributionLoss()(student, teacher, boxes)
            forward_calls = sorter.call_count
            result['loss'].backward()
        self.assertEqual(forward_calls, 4)  # Two teacher and two student view tiles.
        self.assertEqual(sorter.call_count, 6)  # Only the two student tiles repeated.
        # Only pooled [images,M,M] may survive; never [selected_patches,M,M].
        self.assertFalse(any(len(shape) == 3 and shape[-2:] == (3, 3) and shape[0] > 4 for shape in stored))

    def test_reference_cap_excludes_self_and_is_shared_between_branches(self):
        student, teacher, boxes = fixture(52)
        recorded = []
        def capture(views, masks, images, bank, indices):
            recorded.append(indices.clone())
            for image, refs in zip(images.tolist(), indices.tolist()):
                self.assertNotIn(image, refs)
                self.assertEqual(len(set(refs)), MAX_REFERENCE_REGIONS)
            # Avoid expensive sorting in this metadata test.
            value = views[0].sum() * 0 + torch.eye(indices.shape[1])[None].expand(len(images), -1, -1)
            return [value, value]
        with mock.patch('losses.patch_rank_distribution_loss._regional_rank_matrices', capture):
            result = PatchRankDistributionLoss()(student, teacher, boxes)
        torch.testing.assert_close(recorded[0], recorded[1], atol=0, rtol=0)
        self.assertEqual(result['references_per_query'].item(), MAX_REFERENCE_REGIONS)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo required')
    def test_distributed_reference_bank_averaging_and_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=((Path(directory) / 'rendezvous').as_uri(),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
