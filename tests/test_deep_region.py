import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from model.head import iBOTHead
from model.vision_transformer import VisionTransformer
from tests.test_ibot_loss import make_loss
from tests.test_region_loss import boxes_full, boxes_disjoint
from train import get_teacher_targets, load_config
from utils.checkpoint import load_continuation_state, load_resume_state
from utils.register_warmup import teacher_ema_pairs
from utils.training import MultiCropWrapper, get_params_groups


def model(deep=True, registers=0, depth=12, shared=True, wrapped=False, masked=False):
    backbone = VisionTransformer(
        img_size=[32], patch_size=16, embed_dim=8, depth=depth,
        num_heads=2, return_all_tokens=True, num_register_tokens=registers,
        masked_im_modeling=masked,
    )
    head = iBOTHead(8, 3, patch_out_dim=3, hidden_dim=12, bottleneck_dim=4,
                    nlayers=2, shared_head=shared, norm_last_layer=False)
    result = MultiCropWrapper(backbone, head, deep_region=deep)
    if wrapped:
        container = nn.Module()
        container.module = result
        return container
    return result


def distributed_step(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        student = nn.parallel.DistributedDataParallel(model(depth=4, masked=True))
        teacher = model(depth=4)
        teacher.load_state_dict(student.module.state_dict(), strict=False)
        teacher.requires_grad_(False)
        loss = make_loss(region_normalization='deep', region_depths=(1, 2, 3, 4), nlcrops=1)
        optimizer = torch.optim.AdamW(get_params_groups(student), lr=.002)
        for step in range(2):
            torch.manual_seed(32 + rank + step)
            images = [torch.randn(1, 3, 32, 32) for _ in range(2)]
            masks = [torch.tensor([[[True, False], [False, True]]]) for _ in range(2)]
            student_output, student_logits = student(images, mask=masks, return_region_logits=True)
            teacher_output, teacher_logits = teacher(images, return_region_logits=True)
            previous_centers = loss.region_centers.clone()
            targets = get_teacher_targets(teacher_output, loss, 0)
            deep_targets = loss.deep_teacher_targets(teacher_logits, targets[1], .07)
            for index, logits in enumerate(teacher_logits[:-1]):
                expected = logits.mean((0, 1))
                dist.all_reduce(expected)
                torch.testing.assert_close(loss.region_centers[index].flatten(),
                    previous_centers[index].flatten() * .9 + expected / 2 * .1)
            student.module.backbone.masked_im_modeling = False
            local_cls = student(torch.randn(1, 3, 16, 16))[0]
            student.module.backbone.masked_im_modeling = True
            result = loss(student_output, targets, local_cls, masks,
                boxes_full() if rank == 0 else boxes_disjoint(),
                teacher_patch_logits=teacher_output[1], student_region_logits=student_logits,
                teacher_region_logits=teacher_logits, teacher_region_targets=deep_targets)
            optimizer.zero_grad()
            result['loss'].backward()
            for parameter in student.module.region_heads.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            optimizer.step()
            for source, target in teacher_ema_pairs(student.module, teacher):
                target.data.mul_(.9).add_(source.detach(), alpha=.1)
            assert all(parameter.grad is None for parameter in teacher.parameters())
    finally:
        dist.destroy_process_group()


class DeepRegionTest(unittest.TestCase):
    def test_patch_head_copies_match_both_head_layouts_and_are_independent(self):
        for shared in (False, True):
            for bottleneck, layers in ((0, 1), (0, 3), (4, 1), (4, 3)):
                with self.subTest(shared=shared, bottleneck=bottleneck, layers=layers):
                    head = iBOTHead(8, 3, patch_out_dim=5, hidden_dim=12,
                                    bottleneck_dim=bottleneck, nlayers=layers,
                                    shared_head=shared, norm_last_layer=False)
                    projected = head.make_patch_head()
                    tokens = torch.randn(2, 5, 8)
                    torch.testing.assert_close(projected(tokens[:, 1:]), head(tokens)[1])
                    source = head.state_dict()
                    for name, tensor in projected.state_dict().items():
                        original = source[head.patch_head_source_name(name)]
                        torch.testing.assert_close(tensor, original)
                        self.assertNotEqual(tensor.data_ptr(), original.data_ptr())
                    projected(tokens[:, 1:]).square().sum().backward()
                    self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_single_masked_backbone_pass_projects_correct_depths_without_registers(self):
        for depth, expected in ((12, (3, 6, 9, 12)), (24, (6, 12, 18, 24))):
            wrapped = model(registers=4, depth=depth, masked=True).eval()
            self.assertEqual(wrapped.region_layers, expected)
            images = [torch.randn(1, 3, 32, 32) for _ in range(2)]
            masks = [torch.tensor([[[True, False], [False, True]]]) for _ in range(2)]
            captured = {}
            counts = [0] * depth
            def hook(index):
                def capture(_module, _inputs, output):
                    counts[index] += 1
                    captured[index + 1] = wrapped.backbone.norm(output)[:, 5:]
                return capture
            handles = [block.register_forward_hook(hook(i)) for i, block in enumerate(wrapped.backbone.blocks)]
            wrapped.backbone.masked_im_modeling = True
            features, output, logits = wrapped(images, mask=masks,
                return_region_logits=True, return_backbone_feat=True)
            for handle in handles:
                handle.remove()
            self.assertEqual(counts, [1] * depth)
            self.assertEqual(features.shape, (2, 9, 8))
            self.assertIs(logits[-1], output[1])
            for index, block in enumerate(expected[:-1]):
                self.assertEqual(logits[index].shape, (2, 4, 3))
                torch.testing.assert_close(logits[index], wrapped.region_heads[str(block)](captured[block]))
            # Local crops and ordinary callers use only the ordinary output.
            wrapped.backbone.masked_im_modeling = False
            ordinary = wrapped(images)
            _, deep_logits = wrapped(images, return_region_logits=True)
            torch.testing.assert_close(ordinary[1], deep_logits[-1])

    def test_independent_center_targets_use_previous_state_and_reuse_final_ibot_targets(self):
        loss = make_loss(region_normalization='deep', region_depths=(3, 6, 9, 12), center_momentum2=.8)
        loss.region_centers.copy_(torch.arange(9).reshape(3, 1, 1, 3) * .02)
        old = loss.region_centers.clone()
        logits = tuple(torch.randn(4, 4, 3, requires_grad=True) for _ in range(4))
        output = (torch.randn(4, 3), logits[-1])
        targets = get_teacher_targets(output, loss, 0)
        final_center = loss.center2.clone()
        deep_targets = loss.deep_teacher_targets(logits, targets[1], .07)
        self.assertIs(deep_targets[-1], targets[1])
        torch.testing.assert_close(loss.center2, final_center)
        for i in range(3):
            torch.testing.assert_close(deep_targets[i], ((logits[i] - old[i]) / .07).softmax(-1))
            torch.testing.assert_close(loss.region_centers[i], old[i] * .8 + logits[i].mean((0, 1)) * .2)
            self.assertFalse(deep_targets[i].requires_grad)

    def test_region_is_mean_of_four_losses_and_baseline_losses_are_preserved(self):
        for aggregation in ('mean', 'hellinger', 'mean_variance'):
            loss = make_loss(region_normalization='deep', region_depths=(3, 6, 9, 12), region_aggregation=aggregation)
            student_logits = tuple((torch.randn(4, 4, 3) * .1).requires_grad_() for _ in range(4))
            teacher_logits = tuple((torch.randn(4, 4, 3) * .1).requires_grad_() for _ in range(4))
            student = (torch.randn(4, 3, requires_grad=True), student_logits[-1])
            teacher = (torch.randn(4, 3), teacher_logits[-1])
            targets = get_teacher_targets(teacher, loss, 0)
            deep_targets = loss.deep_teacher_targets(teacher_logits, targets[1], .07)
            masks = [torch.ones(2, 2, 2, dtype=torch.bool) for _ in range(2)]
            boxes = boxes_full(2)
            boxes[1] = boxes_disjoint()[0]
            result = loss(student, targets, None, masks, boxes,
                teacher_patch_logits=teacher_logits[-1], student_region_logits=student_logits,
                teacher_region_logits=teacher_logits, teacher_region_targets=deep_targets)
            reference = torch.stack([
                loss.region_loss(s.chunk(2), t.chunk(2), boxes,
                    teacher_patch_targets=q.chunk(2))['loss']
                for s, t, q in zip(student_logits, teacher_logits, deep_targets)
            ]).mean()
            torch.testing.assert_close(result['region_raw'], reference)
            torch.testing.assert_close(result['region'], reference * loss.lambda3)
            baseline = make_loss(lambda3=0)(student, targets, None, masks, None)
            for key in ('cls', 'patch'):
                torch.testing.assert_close(result[key], baseline[key])
            expected_gradients = torch.autograd.grad(reference, student_logits, retain_graph=True)
            result['region_raw'].backward()
            for actual, expected in zip(student_logits, expected_gradients):
                torch.testing.assert_close(actual.grad, expected)
                self.assertGreater(actual.grad.abs().sum(), 0)
            self.assertTrue(all(t.grad is None for t in teacher_logits))

    def test_region_backpropagates_into_independent_intermediate_heads_and_original_final_head(self):
        student = model()
        teacher = model()
        teacher.requires_grad_(False)
        images = [torch.randn(1, 3, 32, 32) for _ in range(2)]
        output, logits = student(images, return_region_logits=True)
        teacher_output, teacher_logits = teacher(images, return_region_logits=True)
        loss = make_loss(region_normalization='deep', region_depths=student.region_layers)
        targets = get_teacher_targets(teacher_output, loss, 0)
        deep_targets = loss.deep_teacher_targets(teacher_logits, targets[1], .07)
        masks = [torch.ones(1, 2, 2, dtype=torch.bool) for _ in range(2)]
        result = loss(output, targets, None, masks, boxes_full(),
            teacher_patch_logits=teacher_output[1], student_region_logits=logits,
            teacher_region_logits=teacher_logits, teacher_region_targets=deep_targets)
        result['region'].backward()
        self.assertTrue(all(p.grad is not None for p in student.head.parameters()))
        for head in student.region_heads.values():
            self.assertTrue(all(p.grad is not None for p in head.parameters()))
            self.assertGreater(sum(p.grad.abs().sum() for p in head.parameters()), 0)
        self.assertTrue(all(t.grad is None for t in teacher.parameters()))

    def test_two_rank_cpu_training_with_empty_overlap_rank_and_local_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(distributed_step, args=(str(Path(directory) / 'gloo'),),
                               nprocs=2, start_method='fork', join=True)

    def test_continuation_copies_pretrained_heads_centers_and_adam_then_resume_restores(self):
        for shared in (True, False):
            source = model(deep=False, shared=shared, wrapped=True)
            source_teacher = model(deep=False, shared=shared)
            optimizer = torch.optim.AdamW(get_params_groups(source))
            image = torch.randn(2, 3, 32, 32)
            sum(x.square().sum() for x in source.module(image)).backward()
            optimizer.step()
            source_loss = make_loss()
            source_loss.center2.fill_(.23)
            checkpoint = {'student': source.state_dict(), 'teacher': source_teacher.state_dict(),
                          'ibot_loss': source_loss.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 2}
            student = model(shared=shared, wrapped=True)
            teacher = model(shared=shared)
            loss = make_loss(region_normalization='deep', region_depths=student.module.region_layers)
            deep_optimizer = torch.optim.AdamW(get_params_groups(student))
            load_continuation_state(checkpoint, student, teacher, loss, deep_optimizer)
            torch.testing.assert_close(loss.region_centers, source_loss.center2.expand_as(loss.region_centers))
            for target, original in ((student.module, source.module), (teacher, source_teacher)):
                tokens = torch.randn(1, 5, 8)
                for head in target.region_heads.values():
                    torch.testing.assert_close(head(tokens[:, 1:]), original.head(tokens)[1])
            source_parameters = dict(source.named_parameters())
            for name, parameter in student.named_parameters():
                if 'region_heads.' in name:
                    suffix = name.split('.', 3)[-1]
                    source_name = 'module.head.' + student.module.head.patch_head_source_name(suffix)
                    for key in ('exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(deep_optimizer.state[parameter][key], optimizer.state[source_parameters[source_name]][key])
            self.assertEqual(len(teacher_ema_pairs(student.module, teacher)), len(list(student.parameters())))
            with torch.no_grad():
                loss.region_centers.copy_(torch.rand_like(loss.region_centers))
                for parameter in student.module.region_heads.parameters():
                    parameter.add_(.01)
            saved = {'student': copy.deepcopy(student.state_dict()), 'teacher': teacher.state_dict(),
                     'optimizer': deep_optimizer.state_dict(), 'ibot_loss': loss.state_dict(), 'epoch': 5}
            resumed = model(shared=shared, wrapped=True)
            resumed_teacher = model(shared=shared)
            resumed_loss = make_loss(region_normalization='deep', region_depths=resumed.module.region_layers)
            resumed_optimizer = torch.optim.AdamW(get_params_groups(resumed))
            self.assertEqual(load_resume_state(saved, resumed, resumed_teacher, resumed_loss, resumed_optimizer, None), 5)
            torch.testing.assert_close(resumed_loss.region_centers, loss.region_centers)
            for name, tensor in student.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], tensor)
            broken = {**saved, 'ibot_loss': {k: v for k, v in saved['ibot_loss'].items() if k != 'region_centers'}}
            with self.assertRaisesRegex(ValueError, 'region_centers'):
                load_resume_state(broken, resumed, resumed_teacher, resumed_loss, resumed_optimizer, None)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                load_resume_state(checkpoint, resumed, resumed_teacher, resumed_loss, resumed_optimizer, None)

    def test_deep_ablation_changes_selector_and_uses_smaller_batch(self):
        root = Path(__file__).parents[1]
        base = (root / 'config/train.yaml').read_bytes()
        ablation = root / 'config/ablations/region_normalization_deep.yaml'
        self.assertEqual(ablation.read_bytes(), base.replace(b'region_normalization: centering', b'region_normalization: deep').replace(b'batch_size_per_gpu: 64', b'batch_size_per_gpu: 32'))
        self.assertEqual(load_config(ablation).region_normalization, 'deep')


if __name__ == '__main__':
    unittest.main()
