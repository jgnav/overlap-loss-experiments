"""CPU coverage of view routing, regional geometry, and baseline isolation."""
import datetime
from itertools import combinations
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from data.augmentations import DataAugmentationiBOT
from data.loader import ImageFolderMask
from losses.local_region_loss import multiview_region_loss
from losses.region_loss import RegionLoss, intersection_patch_fractions
from losses.region_pooling import region_log_probability_mean
from losses.sinkhorn import sinkhorn_log_probabilities
from model.head import iBOTHead
from model.vision_transformer import VisionTransformer
from tests.test_ibot_loss import make_loss
from train import forward_unmasked, get_region_teacher_targets, get_teacher_targets, load_config
from utils.checkpoint import _validate_resume_compatibility
from utils.training import MultiCropWrapper


def fixture():
    torch.manual_seed(72)
    student = tuple((torch.randn(3, 16, 5) * .15).requires_grad_() for _ in range(4))
    teacher = tuple((torch.randn(3, 16, 5) * .15).requires_grad_() for _ in range(4))
    center = torch.linspace(-.1, .1, 5)
    targets = tuple(((x - center) / .07).softmax(-1) for x in teacher)
    full = [0., 0., 1., 1., 0.]
    # Image 0 has six pairs, image 1 one, image 2 none. Flip geometry
    # changes the indices of the selected patches, not the physical overlap.
    boxes = torch.tensor([
        [full, [0., 0., .5, 1., 1.], full, [0., 0., 1., 1., 1.]],
        [[0., 0., .25, .25, 0.]] * 2 + [[.5, .5, .6, .6, 0.], [.9, .9, 1., 1., 0.]],
        [[0., 0., .1, .1, 0.], [.2, .2, .3, .3, 0.],
         [.4, .4, .5, .5, 0.], [.6, .6, .7, .7, 0.]],
    ])
    return student, teacher, targets, boxes


def explicit_reference(s, targets, boxes, min_area=0.):
    """Direct selected-patch mean and CE, no grouped pooling implementation."""
    per_image = []
    for image in range(len(boxes)):
        losses = []
        for a, b in combinations(range(len(s)), 2):
            fractions, positive, _ = intersection_patch_fractions(
                boxes[image:image + 1, [a, b]], (s[a].shape[1], s[b].shape[1]), min_area)
            mask_a, mask_b = (f[0] >= 1 for f in fractions)
            if not positive.item() or not mask_a.any() or not mask_b.any():
                continue
            qa, qb = targets[a][image, mask_a].detach().mean(0), targets[b][image, mask_b].detach().mean(0)
            pa = (s[a][image, mask_a] / .1).softmax(-1).mean(0)
            pb = (s[b][image, mask_b] / .1).softmax(-1).mean(0)
            losses.append(-.5 * ((qa * pb.log()).sum() + (qb * pa.log()).sum()))
        if losses:
            per_image.append(torch.stack(losses).mean())
    return torch.stack(per_image).mean() if per_image else sum(x.sum() * 0 for x in s)


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=30))
    try:
        s, t, targets, boxes = fixture()
        expected = explicit_reference(s, targets, boxes)
        gradients = torch.autograd.grad(expected, s)
        selected = [0, 1] if rank == 0 else [2]
        local_s = tuple(x.detach()[selected].requires_grad_() for x in s)
        objective = RegionLoss(min_area=.1, patch_threshold=1., normalization='centering',
                               temperature=.07, student_temperature=.1)
        result = multiview_region_loss(objective, local_s,
                    tuple(x[selected] for x in t), boxes[selected],
                    teacher_patch_targets=tuple(x[selected] for x in targets))
        result['loss'].backward()
        for actual, gradient in zip(local_s, gradients):
            torch.testing.assert_close(actual.grad / 2, gradient[selected], atol=1e-6, rtol=1e-5)
        averaged = result['loss'].detach().clone()
        dist.all_reduce(averaged)
        torch.testing.assert_close(averaged / 2, expected)
        # The empty rank participates in the one shared SK problem as well.
        objective.normalization = 'sinkhorn'
        result = multiview_region_loss(objective, local_s,
                    tuple(x[selected] for x in t), boxes[selected])
        result['loss'].backward()
        for x in local_s:
            assert torch.isfinite(x.grad).all()
    finally:
        dist.destroy_process_group()


class RegionViewsTest(unittest.TestCase):
    def test_grouped_log_pooling_tiles_preserve_values_and_gradients(self):
        torch.manual_seed(30)
        for dtype in (torch.float32, torch.bfloat16):
            logits = (torch.randn(3, 16, 5) * .2).to(dtype).requires_grad_()
            indices = torch.tensor([0, 2])
            weights = (torch.rand(2, 4, 16) > .5).float()
            weights[:, -1] = 0  # invalid pair slots return connected zeros
            actual = region_log_probability_mean(logits, indices, weights, .1, 16 * 5 * 4)
            logs = (logits[indices].float() / .1).log_softmax(-1)
            expected = torch.stack([
                torch.stack([
                    torch.logsumexp(logs[i, weights[i, r].bool()], 0) - weights[i, r].sum().log()
                    if weights[i, r].any() else logs[i, 0] * 0
                    for r in range(4)]) for i in range(2)])
            upstream = torch.randn_like(expected)
            reference_gradient = torch.autograd.grad(expected, logits, upstream)[0]
            torch.testing.assert_close(actual, expected)
            actual.backward(upstream)
            torch.testing.assert_close(logits.grad, reference_gradient,
                                       atol=.02 if dtype == torch.bfloat16 else 2e-6, rtol=.02)
            self.assertEqual(logits.grad[1].count_nonzero(), 0)

    def test_confident_wrong_predictions_keep_finite_unclamped_ce_gradients(self):
        # exp(-400) underflows in float32; the regional CE must still be 400,
        # with a corrective gradient rather than a probability-clamp plateau.
        s = tuple(torch.tensor([[[20., -20.]] * 4], requires_grad=True) for _ in range(2))
        t = tuple(torch.zeros_like(x) for x in s)
        targets = tuple(torch.tensor([[[0., 1.]] * 4]) for _ in s)
        boxes = torch.tensor([[[0., 0., 1., 1., 0.]] * 2])
        objective = RegionLoss(normalization='centering')
        result = multiview_region_loss(objective, s, t, boxes, teacher_patch_targets=targets)
        self.assertEqual(result['loss'].item(), 400.)
        result['loss'].backward()
        for logits in s:
            torch.testing.assert_close(logits.grad, torch.tensor([[[1.25, -1.25]] * 4]))

    def test_all_pairs_mean_gradient_flips_and_invalid_images(self):
        s, t, targets, boxes = fixture()
        objective = RegionLoss(patch_threshold=1., normalization='centering', temperature=.07)
        result = multiview_region_loss(objective, s, t, boxes, teacher_patch_targets=targets)
        expected = explicit_reference(s, targets, boxes)
        gradients = torch.autograd.grad(expected, s)
        torch.testing.assert_close(result['loss'], expected)
        self.assertAlmostEqual(result['pairs_per_image'].item(), 7 / 3, places=6)
        self.assertAlmostEqual(result['valid_ratio'].item(), 2 / 3, places=6)
        result['loss'].backward()
        for actual, gradient in zip(s, gradients):
            torch.testing.assert_close(actual.grad, gradient, atol=1e-6, rtol=1e-5)
        self.assertTrue(all(x.grad is None for x in t))

    def test_softmax_unique_sinkhorn_and_fp32_under_autocast(self):
        s, t, _, boxes = fixture()
        objective = RegionLoss(normalization='softmax', temperature=.07)
        expected = explicit_reference(s, tuple((x / .07).softmax(-1) for x in t), boxes)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = multiview_region_loss(objective, s, t, boxes)
        self.assertEqual(result['loss'].dtype, torch.float32)
        torch.testing.assert_close(result['loss'], expected)
        objective.normalization = 'sinkhorn'
        with mock.patch('losses.local_region_loss.sinkhorn_log_probabilities',
                        wraps=sinkhorn_log_probabilities) as sinkhorn:
            result = multiview_region_loss(objective, s, t, boxes)
        sinkhorn.assert_called_once()
        # Image 0: 16 cells/view; image 1: only the first two views.
        self.assertEqual(sinkhorn.call_args.args[0].shape, (96, 5))
        self.assertFalse(sinkhorn.call_args.args[0].requires_grad)
        result['loss'].backward()
        self.assertTrue(all(x.grad is None for x in t))

    def test_global_area_filter_and_zero_grad_empty_views(self):
        s, t, targets, boxes = fixture()
        objective = RegionLoss(min_area=.1, normalization='centering', temperature=.07)
        result = multiview_region_loss(objective, s, t, boxes,
                                      teacher_patch_targets=targets, min_area=.1)
        torch.testing.assert_close(result['loss'], explicit_reference(s, targets, boxes, .1))
        self.assertEqual(result['pairs_per_image'].item(), 2.)
        empty_s = tuple(x.detach()[2:].requires_grad_() for x in s)
        for normalization in ('centering', 'sinkhorn'):
            objective.normalization = normalization
            empty = multiview_region_loss(objective, empty_s, tuple(x[2:] for x in t), boxes[2:],
                                          teacher_patch_targets=tuple(x[2:] for x in targets))
            self.assertEqual(empty['loss'].item(), 0)
            empty['loss'].backward()
            self.assertTrue(all(x.grad.count_nonzero() == 0 for x in empty_s))

    def test_extra_augmentation_is_separate_and_preserves_original_crops(self):
        image = Image.fromarray(np.arange(96 * 64 * 3, dtype=np.uint8).reshape(64, 96, 3))
        results = {}
        for mode in ('global', 'global_local', 'local', 'global_unmasked'):
            torch.manual_seed(11)
            random.seed(11)
            results[mode] = DataAugmentationiBOT((.25, 1), (.05, .25), 2, 10, 224, 96,
                                                region_views=mode)(image)
        original, original_boxes = results['global']
        for mode, (images, boxes) in results.items():
            self.assertEqual(len(images), 16 if mode == 'global_unmasked' else 12)
            self.assertEqual(len(boxes), 6 if mode == 'global_unmasked' else (2 if mode == 'global' else 12))
            torch.testing.assert_close(boxes[:2], original_boxes)
            for old, new in zip(original, images):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertTrue(all(x.shape == (3, 224, 224) for x in results['global_unmasked'][0][-4:]))

    def test_dataset_never_masks_the_extra_global_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'class0').mkdir()
            Image.new('RGB', (64, 64), 'orange').save(root / 'class0' / 'image.png')
            dataset = ImageFolderMask(
                root, transform=DataAugmentationiBOT((.25, 1.), (.05, .25), 2, 3, 32, 16,
                                                     region_views='global_unmasked'),
                patch_size=8, pred_ratio=.5, pred_ratio_var=0.,
                pred_aspect_ratio=(.3, 3.), pred_shape='rand')
            images, _, masks, boxes = dataset[0]
            self.assertEqual((len(images), len(masks), len(boxes)), (9, 9, 6))
            self.assertEqual([int(x.sum()) for x in masks], [8, 8, 2, 2, 2, 0, 0, 0, 0])

    def test_real_model_view_routing_and_original_center_updates(self):
        def model(masked):
            return MultiCropWrapper(
                VisionTransformer(img_size=[32], patch_size=8, embed_dim=12,
                                  depth=1, num_heads=3, return_all_tokens=True,
                                  masked_im_modeling=masked, num_register_tokens=2),
                iBOTHead(12, 5, patch_out_dim=5, hidden_dim=16, bottleneck_dim=4,
                         nlayers=2, shared_head=False, norm_last_layer=False))
        for mode in ('global', 'global_local', 'local', 'global_unmasked'):
            with self.subTest(mode=mode):
                torch.manual_seed(3)
                student, teacher = model(True), model(False)
                teacher.load_state_dict(student.state_dict(), strict=False)
                teacher.requires_grad_(False)
                objective = make_loss(out_dim=5, patch_out_dim=5, nlcrops=3,
                                      lambda3=.4, region_views=mode, region_patch_threshold=1.,
                                      region_normalization='centering')
                objective.center2.copy_(torch.tensor([[[.2, -.1, .1, -.2, 0.]]]))
                old_center = objective.center2.clone()
                images = [torch.randn(1, 3, 32, 32) for _ in range(2)]
                images += [torch.randn(1, 3, 16, 16) for _ in range(3)]
                if mode == 'global_unmasked':
                    images += [torch.randn(1, 3, 32, 32) for _ in range(4)]
                masks = [torch.ones(1, 4, 4, dtype=torch.bool) for _ in range(2)]
                seen = {'student': [], 'teacher': []}
                hooks = [network.backbone.register_forward_pre_hook(
                    lambda module, args, name=name: seen[name].append((len(args[0]), module.masked_im_modeling))
                ) for name, network in (('student', student), ('teacher', teacher))]
                try:
                    with torch.no_grad():
                        t_output = teacher(images[:2])
                        extra_logits, extra_targets = get_region_teacher_targets(teacher, images, objective, 0)
                    torch.testing.assert_close(objective.center2, old_center)
                    if extra_logits is not None:
                        torch.testing.assert_close(extra_targets, ((extra_logits - old_center) / .07).softmax(-1))
                    with mock.patch.object(objective, 'update_center', wraps=objective.update_center) as update:
                        targets = get_teacher_targets(t_output, objective, 0)
                    update.assert_called_once()
                    torch.testing.assert_close(objective.center2, old_center * .9 + t_output[1].mean((0, 1), keepdim=True) * .1)
                    s_output = student(images[:2], mask=masks)
                    local_cls, local_logits = forward_unmasked(student, images[2:5])
                    extra_student = (local_logits if mode == 'local' else
                                     forward_unmasked(student, images[-4:])[1] if mode == 'global_unmasked' else None)
                    boxes = torch.tensor([[[0., 0., 1., 1., 0.]] *
                              (6 if mode == 'global_unmasked' else 5 if mode != 'global' else 2)])
                    kwargs = dict(teacher_patch_logits=t_output[1], student_local_patch_logits=local_logits,
                                  student_view_patch_logits=extra_student,
                                  teacher_view_patch_logits=extra_logits, teacher_view_patch_targets=extra_targets)
                    result = objective(s_output, targets, local_cls, masks, boxes, **kwargs)
                    baseline = make_loss(out_dim=5, patch_out_dim=5, nlcrops=3, lambda3=0)
                    control = baseline(s_output, targets, local_cls, masks, None)
                    for name in ('cls', 'patch', 'patch_masked', 'patch_all'):
                        torch.testing.assert_close(result[name], control[name])
                    if mode in ('local', 'global_unmasked'):
                        gradient = torch.autograd.grad(result['region_raw'], s_output[1],
                                                       allow_unused=True, retain_graph=True)[0]
                        self.assertIsNone(gradient)
                        self.assertEqual(result['region_pairs_per_image'].item(), 3 if mode == 'local' else 6)
                    result['loss'].backward()
                    self.assertGreater(student.head.last_layer2.weight_v.grad.abs().sum().item(), 0)
                    self.assertTrue(all(x.grad is None for x in teacher.parameters()))
                    self.assertTrue(student.backbone.masked_im_modeling)
                    self.assertEqual(seen['teacher'], [(2, False)] +
                                     ([(3, False)] if mode == 'local' else [(4, False)] if mode == 'global_unmasked' else []))
                    self.assertEqual(seen['student'], [(2, True), (3, False)] +
                                     ([(4, False)] if mode == 'global_unmasked' else []))
                finally:
                    for hook in hooks:
                        hook.remove()

    def test_disabled_extra_passes_and_mask_mode_restoration(self):
        teacher = mock.Mock(side_effect=AssertionError('Extra teacher forward'))
        for mode in ('local', 'global_unmasked'):
            loss = make_loss(nlcrops=3, region_views=mode, lambda3=0)
            self.assertEqual(get_region_teacher_targets(teacher, [], loss, 0), (None, None))
        network = mock.Mock()
        network.module.backbone.masked_im_modeling = True
        network.side_effect = RuntimeError('test')
        with self.assertRaisesRegex(RuntimeError, 'test'):
            forward_unmasked(network, [])
        self.assertTrue(network.module.backbone.masked_im_modeling)

    def test_launch_configs_only_change_region_views_and_resume_rejects_changed_semantics(self):
        baseline = yaml.safe_load(Path('config/train.yaml').read_text())
        for mode in ('global', 'global_local', 'local', 'global_unmasked'):
            path = Path(f'config/ablations/region_views_{mode}.yaml')
            contents = yaml.safe_load(path.read_text())
            self.assertEqual(contents, {**baseline, 'region_views': mode})
            self.assertEqual(load_config(path).region_views, mode)
        self.assertFalse(Path('config/ablations/include_local_crops_true.yaml').exists())
        with self.assertRaisesRegex(ValueError, 'region_views'):
            _validate_resume_compatibility({'args': SimpleNamespace(region_views='local')},
                                           SimpleNamespace(region_views='global_unmasked', lambda3=0))
        with self.assertRaisesRegex(ValueError, 'old local weighting'):
            _validate_resume_compatibility({'args': SimpleNamespace(include_local_crops=True)},
                                           SimpleNamespace(region_views='global_local', lambda3=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump({**baseline, 'include_local_crops': True}))
            with self.assertRaisesRegex(ValueError, 'Replace include_local_crops'):
                load_config(path)
        with self.assertRaisesRegex(ValueError, 'region_views'):
            make_loss(nlcrops=1, region_views='local')

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo required')
    def test_distributed_normalization_with_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=((Path(directory) / 'rendezvous').as_uri(),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
