import datetime
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from data.augmentations import DataAugmentationiBOT
from losses.local_region_loss import global_local_region_loss
from losses.region_loss import RegionLoss, intersection_patch_fractions
from losses.sinkhorn import sinkhorn_log_probabilities
from model.head import iBOTHead
from model.vision_transformer import VisionTransformer
from tests.test_ibot_loss import make_loss
from train import load_config
from utils.checkpoint import _validate_resume_compatibility
from utils.training import MultiCropWrapper


def inputs():
    torch.manual_seed(19)
    globals_s = tuple((torch.randn(4, 16, 5) * .1).requires_grad_() for _ in range(2))
    globals_t = tuple((torch.randn(4, 16, 5) * .1).requires_grad_() for _ in range(2))
    locals_s = tuple((torch.randn(4, 4, 5) * .1).requires_grad_() for _ in range(3))
    center = torch.linspace(-.1, .1, 5)
    targets = tuple(((x - center) / .07).softmax(-1) for x in globals_t)
    full = [0., 0., 1., 1., 0.]
    upper = [0., 0., .5, .5, 0.]
    lower = [.5, .5, 1., 1., 0.]
    tiny = [.45, .45, .55, .55, 0.]
    boxes = torch.tensor([
        [[0., 0., 1., 1., 1.], full, full, [0., 0., .5, 1., 0.], [.05, .05, .1, .1, 0.]],
        [upper, upper, [.75, .75, 1., 1., 0.], lower, lower],
        [upper, lower, upper, lower, tiny],
        [upper, lower, tiny, tiny, tiny],
    ])
    return globals_s, globals_t, locals_s, targets, boxes


def evaluate(s, t, local, targets, boxes, mode="centering"):
    objective = RegionLoss(min_area=.1, patch_threshold=1., temperature=.07,
                           normalization=mode, student_temperature=.1)
    global_stats = objective(s, t, boxes[:, :2], teacher_patch_targets=targets,
                             return_per_image=True)
    return global_local_region_loss(objective, global_stats, local, t, boxes,
                                    teacher_patch_targets=targets)


def reference(s, local, targets):
    # Hand-selected cells for the controlled geometry, independent of the loss
    # implementation. Different numbers of valid local pairs must not change
    # the weighting of images or the global/global group.
    def ce(t, prediction):
        return -(t.detach().mean(0) * (prediction / .1).softmax(-1).mean(0).log()).sum()

    gg = [.5 * (ce(targets[0][i], s[1][i]) + ce(targets[1][i], s[0][i])) for i in (0, 1)]
    right = [2, 3, 6, 7, 10, 11, 14, 15]
    left = [0, 1, 4, 5, 8, 9, 12, 13]
    gl0 = torch.stack([
        ce(targets[0][0], local[0][0]), ce(targets[1][0], local[0][0]),
        ce(targets[0][0, right], local[1][0]), ce(targets[1][0, left], local[1][0]),
    ]).mean()
    gl2 = .5 * (ce(targets[0][2], local[0][2]) + ce(targets[1][2], local[1][2]))
    return (torch.stack([.75 * gg[0] + .25 * gl0, gg[1], gl2]).mean()
            + sum(x.sum() * 0 for x in local))


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=30))
    try:
        s, t, local, targets, boxes = inputs()
        expected = reference(s, local, targets)
        indices = [0, 1, 2] if rank == 0 else [3]  # one rank has no valid groups
        local_s = tuple(x.detach()[indices].requires_grad_() for x in s)
        local_l = tuple(x.detach()[indices].requires_grad_() for x in local)
        for mode in ("centering", "sinkhorn"):
            result = evaluate(local_s, tuple(x[indices] for x in t), local_l,
                              tuple(x[indices] for x in targets), boxes[indices], mode)
            result["loss"].backward()
            averaged = result["loss"].detach().clone()
            dist.all_reduce(averaged)
            if mode == "centering":
                torch.testing.assert_close(averaged / 2, expected)
            for x in (*local_s, *local_l):
                assert torch.isfinite(x.grad).all()
                if rank == 1:
                    assert x.grad.count_nonzero() == 0
                x.grad = None
        # All-empty intersections still permit backward, including SK collectives.
        boxes[:, 2:] = torch.tensor([.45, .45, .55, .55, 0.])
        boxes[:, :2] = boxes[3, :2].clone()
        result = evaluate(local_s, tuple(x[indices] for x in t), local_l,
                          tuple(x[indices] for x in targets), boxes[indices], "sinkhorn")
        assert result["loss"].item() == 0
        result["loss"].backward()
    finally:
        dist.destroy_process_group()


class LocalRegionTest(unittest.TestCase):
    def test_per_image_means_mixing_fallbacks_and_gradients(self):
        s, t, local, targets, boxes = inputs()
        result = evaluate(s, t, local, targets, boxes)
        expected = reference(s, local, targets)
        gradients = torch.autograd.grad(expected, (*s, *local))
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(result["valid_ratio"].item(), .75)
        self.assertEqual(result["global_local_valid_ratio"].item(), .5)
        self.assertEqual(result["global_local_pairs_per_image"].item(), 1.5)
        result["loss"].backward()
        for x, gradient in zip((*s, *local), gradients):
            torch.testing.assert_close(x.grad, gradient, atol=1e-6, rtol=1e-5)
        self.assertTrue(all(x.grad is None for x in t))
        self.assertTrue(all(x.grad[1].count_nonzero() == 0 for x in local))
        self.assertEqual(local[2].grad.count_nonzero(), 0)

    def test_full_containment_at_native_grids_with_flips(self):
        boxes = torch.tensor([[[0., 0., 1., 1., 1.], [0., 0., .5, 1., 1.]]])
        fractions, valid, _ = intersection_patch_fractions(boxes, (16, 4), 0)
        self.assertTrue(valid.item())
        torch.testing.assert_close((fractions[0] >= 1)[0], torch.tensor(
            [False, False, True, True] * 4
        ))
        self.assertTrue((fractions[1] == 1).all())
        # Teacher-native 14x14 and local-native 6x6 grids accept all cells for
        # an identical crop even though normalized edges round in float32.
        boxes[:, 1] = boxes[:, 0]
        fractions, _, _ = intersection_patch_fractions(boxes, (196, 36), 0)
        self.assertTrue(all((fraction == 1).all() for fraction in fractions))

    def test_student_local_boundary_patches_are_excluded(self):
        s = tuple(torch.zeros(1, 16, 3, requires_grad=True) for _ in range(2))
        t = tuple(torch.zeros_like(x) for x in s)
        local = (torch.randn(1, 16, 3, requires_grad=True),)
        boxes = torch.tensor([[
            [0., 0., .625, 1., 0.], [.8, 0., 1., 1., 0.], [0., 0., 1., 1., 1.]
        ]])
        targets = tuple(x.softmax(-1) for x in t)
        result = evaluate(s, t, local, targets, boxes)
        selected = torch.tensor([False, False, True, True] * 4)
        expected = -(targets[0][0].mean(0) * (local[0][0, selected] / .1).softmax(-1).mean(0).log()).sum()
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        self.assertEqual(local[0].grad[0, ~selected].count_nonzero(), 0)
        self.assertGreater(local[0].grad[0, selected].abs().sum().item(), 0)

    def test_default_false_and_disabled_region_leave_baseline_losses_unchanged(self):
        s, t, local, targets, boxes = inputs()
        global_student = (torch.randn(8, 5, requires_grad=True), torch.cat(s))
        local_cls = torch.randn(12, 5, requires_grad=True)
        teacher_targets = (torch.randn(8, 5).softmax(-1), torch.cat(targets))
        masks = [torch.ones(4, 4, 4, dtype=torch.bool) for _ in range(2)]
        arguments = dict(out_dim=5, patch_out_dim=5, nlcrops=3, lambda3=.4,
                         region_min_area=.1, region_patch_threshold=1., region_normalization="centering")
        baseline = make_loss(**arguments)
        active = make_loss(**arguments, include_local_crops=True)
        a = baseline(global_student, teacher_targets, local_cls, masks, boxes[:, :2],
                     teacher_patch_logits=torch.cat(t))
        b = active(global_student, teacher_targets, local_cls, masks, boxes,
                   teacher_patch_logits=torch.cat(t), student_local_patch_logits=torch.cat(local))
        for name in ("cls", "patch", "patch_masked", "patch_all"):
            torch.testing.assert_close(a[name], b[name])
        torch.testing.assert_close(b["region_raw"], reference(s, local, targets))
        torch.testing.assert_close(b["region"], b["region_raw"] * .4)
        torch.testing.assert_close(a["region_raw"], baseline.region_loss(
            s, t, boxes[:, :2], teacher_patch_targets=targets)["loss"])
        disabled = make_loss(**{**arguments, "lambda3": 0}, include_local_crops=True)
        with mock.patch("losses.ibot_loss.global_local_region_loss", side_effect=AssertionError):
            result = disabled(global_student, teacher_targets, local_cls, masks, None)
        torch.testing.assert_close(result["loss"], result["cls"] + result["patch"])

    def test_softmax_reference_and_unique_sinkhorn_teacher_assignments(self):
        s, t, local, targets, boxes = inputs()
        uncentered = tuple((x / .07).softmax(-1) for x in t)
        result = evaluate(s, t, local, targets, boxes, "softmax")
        torch.testing.assert_close(result["loss"], reference(s, local, uncentered))
        with mock.patch("losses.local_region_loss.sinkhorn_log_probabilities",
                        wraps=sinkhorn_log_probabilities) as sinkhorn:
            result = evaluate(s, t, local, targets, boxes, "sinkhorn")
        sinkhorn.assert_called_once()
        selected = sinkhorn.call_args.args[0]
        self.assertEqual(selected.shape, (64, 5))
        # Two eligible images, two teacher views, sixteen unique patches each.
        expected = torch.stack(t, 1)[[0, 2]].flatten(0, 2)
        torch.testing.assert_close(selected, expected)
        self.assertFalse(selected.requires_grad)
        result["loss"].backward()
        self.assertTrue(all(x.grad is None for x in t))

    def test_actual_teacher_global_and_student_all_crop_forwards(self):
        def model(masked):
            return MultiCropWrapper(
                VisionTransformer(img_size=[224], patch_size=16, embed_dim=12,
                                  depth=1, num_heads=3, return_all_tokens=True,
                                  masked_im_modeling=masked, num_register_tokens=2),
                iBOTHead(12, 5, patch_out_dim=5, hidden_dim=16, bottleneck_dim=4,
                         nlayers=2, shared_head=False, norm_last_layer=False),
            )

        torch.manual_seed(32)
        student, teacher = model(True), model(False)
        teacher.load_state_dict(student.state_dict(), strict=False)
        teacher.requires_grad_(False)
        seen = {"student": [], "teacher": []}
        hooks = [network.backbone.register_forward_pre_hook(
            lambda module, args, name=name: seen[name].append(args[0].shape)
        ) for name, network in (("student", student), ("teacher", teacher))]
        try:
            images = [torch.randn(1, 3, 224, 224) for _ in range(2)]
            images += [torch.randn(1, 3, 96, 96) for _ in range(10)]
            masks = [torch.ones(1, 14, 14, dtype=torch.bool) for _ in range(2)]
            with torch.no_grad():
                teacher_output = teacher(images[:2])
            student_output = student(images[:2], mask=masks)
            student.backbone.masked_im_modeling = False
            local_cls, local_patch = student(images[2:])
            student.backbone.masked_im_modeling = True
            self.assertEqual(seen["teacher"], [torch.Size([2, 3, 224, 224])])
            self.assertEqual(seen["student"], [torch.Size([2, 3, 224, 224]), torch.Size([10, 3, 96, 96])])
            self.assertEqual(local_patch.shape, (10, 36, 5))
            objective = make_loss(out_dim=5, patch_out_dim=5, nlcrops=10, lambda1=0,
                                  lambda2=0, lambda3=.4, include_local_crops=True,
                                  region_patch_threshold=1., region_normalization="centering")
            targets = objective.softmax_center_teacher(teacher_output, .07, .07)
            a = [0., 0., .5, .5, 0.]
            b = [.5, .5, 1., 1., 0.]
            boxes = torch.tensor([[a, b] + [a, b] * 5])
            result = objective(student_output, targets, local_cls, masks, boxes,
                               teacher_patch_logits=teacher_output[1],
                               student_local_patch_logits=local_patch)
            self.assertEqual(result["region_global_local_pairs_per_image"].item(), 10)
            # Global/global is invalid, so the regional gradient comes solely
            # from the local patch head and local backbone representations.
            self.assertEqual(result["region_global_global_loss"].item(), 0)
            result["loss"].backward()
            self.assertGreater(student.head.last_layer2.weight_v.grad.abs().sum().item(), 0)
            self.assertGreater(student.backbone.patch_embed.proj.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(x.grad is None for x in teacher.parameters()))
        finally:
            for hook in hooks:
                hook.remove()

    def test_augmentation_preserves_crops_and_records_local_geometry_only_when_enabled(self):
        image = Image.fromarray(np.arange(96 * 64 * 3, dtype=np.uint8).reshape(64, 96, 3))
        outputs = []
        for enabled in (False, True):
            random.seed(11)
            torch.manual_seed(11)
            augmentation = DataAugmentationiBOT((.25, 1.), (.05, .25), 2, 10, 224, 96,
                                                include_local_crops=enabled)
            with mock.patch.object(augmentation, "_spatial_transform",
                                   wraps=augmentation._spatial_transform) as spatial:
                outputs.append(augmentation(image))
                self.assertEqual(spatial.call_count, 12)
        a, b = outputs
        self.assertEqual(a[1].shape, (2, 5))
        self.assertEqual(b[1].shape, (12, 5))
        torch.testing.assert_close(a[1], b[1][:2])
        for first, second in zip(a[0], b[0]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(((b[1][:, :4] >= 0) & (b[1][:, :4] <= 1)).all())

    def test_yaml_ablation_and_resume_validation(self):
        base_path = Path("config/train.yaml")
        ablation_path = Path("config/ablations/include_local_crops_true.yaml")
        self.assertEqual(ablation_path.read_bytes(), base_path.read_bytes().replace(
            b"include_local_crops: false", b"include_local_crops: true"
        ))
        config = load_config(ablation_path)
        self.assertTrue(config.include_local_crops)
        self.assertEqual((config.lambda3, config.region_aggregation), (.4, "mean"))
        self.assertEqual((config.global_crops_number, config.local_crops_number), (2, 10))
        base = yaml.safe_load(base_path.read_text())
        for invalid in ("true", 1, None):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "train.yaml"
                path.write_text(yaml.safe_dump({**base, "include_local_crops": invalid}))
                with self.assertRaisesRegex(ValueError, "include_local_crops"):
                    load_config(path)
        with self.assertRaisesRegex(ValueError, "include_local_crops"):
            _validate_resume_compatibility({"args": SimpleNamespace(include_local_crops=False)},
                                           SimpleNamespace(include_local_crops=True, lambda3=0))
        with self.assertRaisesRegex(ValueError, "include_local_crops"):
            _validate_resume_compatibility({"args": SimpleNamespace()},
                                           SimpleNamespace(include_local_crops=True, lambda3=0))
        for kwargs in ({"region_aggregation": "hellinger"}, {"region_normalization": "deep"},
                       {"region_normalization": "raw_logits"}, {"nlcrops": 0}):
            with self.assertRaisesRegex(ValueError, "include_local_crops"):
                make_loss(**{"nlcrops": 3, "include_local_crops": True, **kwargs})

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_distributed_valid_image_normalization_and_empty_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=((Path(directory) / "rendezvous").as_uri(),),
                     nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
