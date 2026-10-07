import copy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import yaml

from losses.region_token_loss import RegionTokenLoss
from losses.sinkhorn import sinkhorn_log_probabilities
from model.head import iBOTHead
from model.region_token import RegionTokenAggregation
from model.vision_transformer import VisionTransformer
from tests.test_ibot_loss import make_loss
from tests.test_region_loss import boxes_full, boxes_disjoint
from train import get_teacher_targets, load_config
from utils.checkpoint import load_continuation_state, load_resume_state, _validate_resume_compatibility
from utils.register_warmup import teacher_ema_pairs
from utils.training import MultiCropWrapper, get_params_groups


def model(token=True, masked=False, registers=0, shared=True):
    backbone = VisionTransformer(
        img_size=[32], patch_size=16, embed_dim=8, depth=2, num_heads=2,
        return_all_tokens=True, masked_im_modeling=masked,
        num_register_tokens=registers,
    )
    head = iBOTHead(8, 3, patch_out_dim=5, hidden_dim=12, bottleneck_dim=4,
                    nlayers=2, shared_head=shared, norm_last_layer=False)
    return MultiCropWrapper(backbone, head, region_token=token)


def step(student, teacher, loss, boxes, local=False):
    images = [torch.randn(len(boxes), 3, 32, 32) for _ in range(2)]
    masks = [torch.tensor([[[True, False], [False, True]]]).expand(len(boxes), -1, -1)
             for _ in range(2)]
    weights = list(loss.region_loss.prepare_geometry(boxes, 4)['weights'].unbind(1))
    with torch.no_grad():
        teacher_output, teacher_regions = teacher(images, region_weights=weights)
        targets = get_teacher_targets(teacher_output, loss, 0)
    student_output, student_regions = student(images, mask=masks, region_weights=weights)
    local_cls = None
    if local:
        wrapper = student.module if hasattr(student, 'module') else student
        wrapper.backbone.masked_im_modeling = False
        local_cls = student(torch.randn(len(boxes), 3, 16, 16))[0]
        wrapper.backbone.masked_im_modeling = True
    return loss(student_output, targets, local_cls, masks, boxes,
                teacher_patch_logits=teacher_output[1],
                student_region_logits=student_regions, teacher_region_logits=teacher_regions)


def distributed_step(rank, rendezvous, normalization):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        student = nn.parallel.DistributedDataParallel(model(masked=True))
        teacher = model().requires_grad_(False)
        teacher.load_state_dict(student.module.state_dict(), strict=False)
        loss = make_loss(region_aggregation='region_token', region_normalization=normalization, nlcrops=1)
        optimizer = torch.optim.AdamW(get_params_groups(student), lr=.002)
        for index in range(2):
            torch.manual_seed(100 + rank + index)
            boxes = boxes_full() if rank == 0 and index == 0 else boxes_disjoint()
            old_center = loss.region_loss.center.clone() if normalization == 'centering' else None
            result = step(student, teacher, loss, boxes, local=True)
            optimizer.zero_grad()
            result['loss'].backward()
            for parameter in student.module.region_token.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                if index == 1:
                    assert parameter.grad.count_nonzero() == 0
            if normalization == 'centering':
                centers = [torch.empty_like(loss.region_loss.center) for _ in range(2)]
                dist.all_gather(centers, loss.region_loss.center)
                torch.testing.assert_close(centers[0], centers[1])
            if index == 1:
                assert result['region_raw'].item() == 0
                if normalization == 'centering':
                    torch.testing.assert_close(loss.region_loss.center, old_center)
            optimizer.step()
            for source, target in teacher_ema_pairs(student.module, teacher):
                target.data.mul_(.9).add_(source.detach(), alpha=.1)
            assert all(parameter.grad is None for parameter in teacher.parameters())
    finally:
        dist.destroy_process_group()


class RegionTokenTest(unittest.TestCase):
    def test_attention_matches_masked_concept_row_and_excludes_other_patches(self):
        torch.manual_seed(3)
        aggregate = RegionTokenAggregation(8, 2)
        patches = torch.randn(2, 4, 8, requires_grad=True)
        weights = torch.tensor([[1., 0., .25, 0.], [0., 0., 0., 0.]])
        actual = aggregate(patches, weights)
        # Independent dense-mask reference: only concept row has patch keys.
        sequence = torch.cat((aggregate.token, patches[:1]), dim=1)
        qkv = aggregate.block.attn.qkv(aggregate.block.norm1(sequence)).reshape(1, 5, 3, 2, 4)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) * .5
        bias = torch.tensor([-torch.inf, 0., -torch.inf, .25, -torch.inf])
        bias[3] = torch.tensor(.25).log()
        probabilities = (scores[:, :, :1] + bias).softmax(-1)
        pooled = (probabilities @ v).transpose(1, 2).reshape(1, 1, 8)
        token = aggregate.token + aggregate.block.attn.proj(pooled)
        token = token + aggregate.block.mlp(aggregate.block.norm2(token))
        expected = aggregate.norm(token[:, 0])
        torch.testing.assert_close(actual[:1], expected)
        self.assertEqual(actual[1].count_nonzero(), 0)
        changed = patches.detach().clone()
        changed[weights == 0] = torch.nan
        torch.testing.assert_close(actual, aggregate(changed, weights))
        permutation = torch.tensor([2, 3, 0, 1])
        torch.testing.assert_close(actual, aggregate(patches[:, permutation], weights[:, permutation]))
        actual.square().sum().backward()
        self.assertEqual(patches.grad[weights == 0].count_nonzero(), 0)
        self.assertGreater(aggregate.token.grad.abs().sum(), 0)
        self.assertTrue(torch.isfinite(patches.grad).all())

    def test_region_distillation_formula_center_and_empty_pairs(self):
        torch.manual_seed(7)
        loss = RegionTokenLoss(normalization='centering', aggregation='region_token',
                               out_dim=3, temperature=.07, student_temperature=.1)
        loss.center.copy_(torch.tensor([[.01, -.02, .03]]))
        old_center = loss.center.clone()
        s = tuple(torch.randn(2, 3, requires_grad=True) for _ in range(2))
        t = tuple(torch.randn(2, 3, requires_grad=True) for _ in range(2))
        boxes = boxes_full(2)
        boxes[1:] = boxes_disjoint()
        result = loss(s, t, boxes, patch_count=4)
        expected = -.5 * sum(
            (((t[view][0].detach() - old_center[0]) / .07).softmax(-1)
             * (s[1 - view][0] / .1).log_softmax(-1)).sum()
            for view in range(2)
        )
        torch.testing.assert_close(result['loss'], expected)
        torch.testing.assert_close(loss.center, old_center * .9 +
                                   (t[0][0].detach() + t[1][0].detach())[None] * .05)
        result['loss'].backward()
        self.assertTrue(all(x.grad[1].count_nonzero() == 0 for x in s))
        self.assertTrue(all(x.grad is None for x in t))
        old_center = loss.center.clone()
        empty = loss(s, t, boxes_disjoint(2), patch_count=4)
        self.assertEqual(empty['loss'].item(), 0)
        empty['loss'].backward()
        torch.testing.assert_close(loss.center, old_center)

    def test_projection_sharing_register_exclusion_and_single_backbone_pass(self):
        for shared in (False, True):
            wrapped = model(registers=2, shared=shared).eval()
            images = [torch.randn(1, 3, 32, 32) for _ in range(2)]
            weights = [torch.tensor([[1., 0., 1., 0.]]) for _ in range(2)]
            counts = [0]
            handle = wrapped.backbone.register_forward_hook(lambda *_args: counts.__setitem__(0, counts[0] + 1))
            features, output, regions = wrapped(images, region_weights=weights, return_backbone_feat=True)
            handle.remove()
            self.assertEqual(counts[0], 1)
            torch.testing.assert_close(regions, wrapped.head.project_patches(
                wrapped.region_token(features[:, 3:], torch.cat(weights))))
            ordinary = wrapped(images)
            for actual, expected in zip(output, ordinary):
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(wrapped.head.project_patches(features[:, 3:]), output[1])
            self.assertEqual(regions.shape, (2, 3 if shared else 5))

    def test_softmax_and_teacher_only_sinkhorn_tokens_and_mixed_precision(self):
        for mode in ('softmax', 'sinkhorn'):
            torch.manual_seed(8)
            s = tuple(torch.randn(2, 3, requires_grad=True) * .1 for _ in range(2))
            t = tuple(torch.randn(2, 3, requires_grad=True) * .1 for _ in range(2))
            loss = RegionTokenLoss(normalization=mode, aggregation='region_token', out_dim=3)
            result = loss(s, t, boxes_full(2), patch_count=4)
            teacher = torch.stack(t, 1).detach()
            targets = ((teacher / .1).softmax(-1) if mode == 'softmax' else
                       sinkhorn_log_probabilities(teacher.flatten(0, 1), .1).exp().reshape_as(teacher))
            log_probs = (torch.stack(s, 1) / .1).log_softmax(-1)
            expected = -.5 * ((targets[:, 0] * log_probs[:, 1]).sum(-1)
                              + (targets[:, 1] * log_probs[:, 0]).sum(-1)).mean()
            torch.testing.assert_close(result['loss'], expected)
            gradients = torch.autograd.grad(result['loss'], (*s, *t), allow_unused=True)
            self.assertTrue(all(x is not None and torch.isfinite(x).all() for x in gradients[:2]))
            self.assertEqual(gradients[2:], (None, None))
        aggregate = RegionTokenAggregation(8, 2)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            features = aggregate(torch.randn(2, 4, 8), torch.tensor([[1., 0., 1., 0.], [0., 0., 0., 0.]]))
        features.square().sum().backward()
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in aggregate.parameters()))

    def test_continuation_initialization_and_full_resume_restore_all_state(self):
        torch.manual_seed(9)
        original = model(token=False, masked=True)
        source = {'student': original.state_dict(),
                  'teacher': {name: value for name, value in original.state_dict().items()
                              if not name.endswith('masked_embed')},
                  'ibot_loss': make_loss().state_dict()}
        student, teacher = model(masked=True), model()
        loss = make_loss(region_aggregation='region_token', region_normalization='centering')
        optimizer = torch.optim.AdamW(get_params_groups(student), lr=.002)
        load_continuation_state(source, student, teacher, loss, optimizer, reset_optimizer=True)
        for name, value in student.region_token.state_dict().items():
            torch.testing.assert_close(value, teacher.region_token.state_dict()[name])
        teacher.requires_grad_(False)
        result = step(student, teacher, loss, boxes_full())
        result['loss'].backward()
        optimizer.step()
        for source_parameter, target in teacher_ema_pairs(student, teacher):
            target.data.mul_(.9).add_(source_parameter.detach(), alpha=.1)
        checkpoint = copy.deepcopy({'student': student.state_dict(), 'teacher': teacher.state_dict(),
                                    'ibot_loss': loss.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 1})
        restored_s, restored_t = model(masked=True), model()
        restored_loss = make_loss(region_aggregation='region_token', region_normalization='centering')
        restored_optimizer = torch.optim.AdamW(get_params_groups(restored_s), lr=.002)
        self.assertEqual(load_resume_state(checkpoint, restored_s, restored_t, restored_loss, restored_optimizer, None), 1)
        for actual, expected in ((restored_s, student), (restored_t, teacher), (restored_loss, loss)):
            for name, value in actual.state_dict().items():
                torch.testing.assert_close(value, expected.state_dict()[name])
        for key, state in optimizer.state_dict()['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, restored_optimizer.state_dict()['state'][key][name])
        torch.manual_seed(40)
        expected = step(student, teacher, loss, boxes_full())['loss']
        torch.manual_seed(40)
        actual = step(restored_s, restored_t, restored_loss, boxes_full())['loss']
        torch.testing.assert_close(actual, expected)
        broken = copy.deepcopy(checkpoint)
        del broken['ibot_loss']['region_loss.center']
        with self.assertRaisesRegex(ValueError, 'independent center'):
            load_resume_state(broken, restored_s, restored_t, restored_loss, restored_optimizer, None)

    def test_config_changes_only_aggregation_and_rejects_incompatible_combinations(self):
        root = Path(__file__).parents[1]
        default = yaml.safe_load((root / 'config/train_ablation.yaml').read_text())
        path = root / 'config/ablations/region_aggregation_region_token.yaml'
        ablation = yaml.safe_load(path.read_text())
        self.assertEqual({key for key in default if default[key] != ablation[key]}, {'region_aggregation'})
        self.assertEqual(load_config(path).region_aggregation, 'region_token')
        for options in (dict(region_normalization='deep'), dict(region_normalization='raw_logits'),
                        dict(region_views='global_local'), dict(loss_modality='within_image')):
            with self.assertRaises(ValueError):
                make_loss(region_aggregation='region_token', **options)
        with self.assertRaisesRegex(ValueError, 'region_aggregation'):
            _validate_resume_compatibility({'args': {'region_aggregation': 'mean'}},
                                          type('Args', (), {'region_aggregation': 'region_token'})())

    def test_distributed_empty_rank_all_empty_batch_local_forward_and_ema(self):
        for mode in ('centering', 'sinkhorn'):
            with self.subTest(normalization=mode), tempfile.TemporaryDirectory() as directory:
                mp.spawn(distributed_step, args=(str(Path(directory) / 'rendezvous'), mode), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
