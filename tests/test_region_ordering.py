import copy
import datetime
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from itertools import combinations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from losses.region_ordering_loss import (
    ORDERING_WEIGHT, RegionOrderingLoss, collect_overlap_regions,
    sample_external_references, _compare_queries, _contained_patch_masks,
)
from losses.region_sorting import bitonic_permutation, permutation_cross_entropy
from losses.region_pooling import region_probability_mean
from losses.region_loss import intersection_patch_fractions
from tests.test_ibot_loss import make_loss
from train import load_config
from utils.checkpoint import _validate_resume_compatibility, load_pretrained_state, load_resume_state
from model.head import iBOTHead
from model.vision_transformer import VisionTransformer
from utils.training import MultiCropWrapper


def fixture(batch=2):
    torch.manual_seed(173)
    boxes = torch.tensor([[
        [0, 0, 1, 1, 0], [0, 0, 1, 1, 1],
        [0, 0, .5, .5, 0], [.5, .5, 1, 1, 1], [.25, .25, .75, .75, 0],
    ]] * batch, dtype=torch.float32)
    student = tuple((torch.randn(batch, patches, 7) * .03).requires_grad_()
                    for patches in (64, 64, 16, 16, 16))
    teacher = tuple((torch.randn(batch, 64, 7) * .04).requires_grad_() for _ in range(2))
    targets = tuple(((view - torch.linspace(-.02, .02, 7)) / .07).softmax(-1) for view in teacher)
    return student, teacher, targets, boxes


def cells(grid, rows, columns):
    return [row * grid + column for row in rows for column in columns]


def manual_within_loss(student, targets):
    # Six physical regions: full, top-left, bottom-right, center,
    # top-left/center intersection, bottom-right/center intersection.
    selections = [list(range(64)), cells(8, range(4), range(4)),
                  cells(8, range(4, 8), range(4, 8)), cells(8, range(2, 6), range(2, 6)),
                  cells(8, range(2, 4), range(2, 4)), cells(8, range(4, 6), range(4, 6))]
    queries = [(0, 0, 1, 0, list(range(64))), (0, 1, 0, 0, list(range(64))),
               (1, 0, 2, 1, list(range(16))), (2, 0, 3, 1, list(range(16))),
               (3, 0, 4, 1, list(range(16))),
               (1, 1, 2, 1, list(range(16))), (2, 1, 3, 1, list(range(16))),
               (3, 1, 4, 1, list(range(16))),
               (4, 0, 2, 2, cells(4, range(2, 4), range(2, 4))),
               (4, 0, 4, 2, cells(4, range(2), range(2))),
               (5, 0, 3, 2, cells(4, range(2), range(2, 4))),
               (5, 0, 4, 2, cells(4, range(2, 4), range(2, 4)))]
    families = [[], [], []]
    for image in range(len(student[0])):
        bank = torch.stack([targets[0][image, selected].detach().mean(0) for selected in selections])
        for region, teacher_view, student_view, family, selected in queries:
            teacher_selected = selections[region]
            if teacher_view == 1:
                teacher_selected = [index // 8 * 8 + 7 - index % 8 for index in teacher_selected]
            teacher = targets[teacher_view][image, teacher_selected].detach().mean(0)
            prediction = (student[student_view][image, selected] / .1).softmax(-1).mean(0)
            references = F.normalize(bank[[index for index in range(6) if index != region]], dim=-1)
            families[family].append(permutation_cross_entropy(
                (F.normalize(prediction, dim=0) @ references.T)[None],
                (F.normalize(teacher, dim=0) @ references.T)[None],
            ).squeeze())
    means = torch.stack([torch.stack(family).mean() for family in families])
    return (means * means.new_tensor([.5, .25, .25])).sum()


def manual_cross_loss(student, targets):
    # One physical overlap per image, two cross-view queries. Both full global
    # views have the same patch count, so global 0 supplies each reference.
    bank = torch.stack([targets[0][image].detach().mean(0) for image in range(len(student[0]))])
    losses = []
    for image in range(len(student[0])):
        references = F.normalize(bank[[other for other in range(len(bank)) if other != image]], dim=-1)
        for teacher_view, student_view in ((0, 1), (1, 0)):
            teacher = targets[teacher_view][image].detach().mean(0)
            prediction = (student[student_view][image] / .1).softmax(-1).mean(0)
            losses.append(permutation_cross_entropy(
                (F.normalize(prediction, dim=0) @ references.T)[None],
                (F.normalize(teacher, dim=0) @ references.T)[None],
            ).squeeze())
    return torch.stack(losses).mean()


def invalidate_geometry(boxes):
    boxes[:, 0, :4] = torch.tensor([0., 0., .1, .1])
    boxes[:, 1, :4] = torch.tensor([.9, .9, 1., 1.])
    boxes[:, 2:, :4] = torch.tensor([.45, .45, .46, .46])


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        # Different prototype predictions for each image/rank, geometry fixed.
        all_student, _, targets, boxes = fixture(4)
        subset = slice(rank * 2, rank * 2 + 2)
        for mode in ("within_image", "cross_image"):
            distributed_student = tuple(view[subset].detach().requires_grad_() for view in all_student)
            # Hold reference input order fixed for the independent gradient
            # calculation: finite-temperature sorting depends on wire order.
            with mock.patch("losses.region_ordering_loss.sample_external_references",
                            side_effect=lambda rows, owner, count, generator: [
                                index for other, indices in rows.items() if other != owner for index in indices
                            ]):
                result = RegionOrderingLoss(mode)(distributed_student,
                                                  tuple(view[subset] for view in targets), boxes[subset])
            assert result["query_count"].item() == (48 if mode == "within_image" else 8)
            assert result["references_per_query"].item() == (5 if mode == "within_image" else 3)
            assert result["reference_region_count"].item() == (24 if mode == "within_image" else 4)
            result["loss"].backward()
            assert all(torch.isfinite(view.grad).all() for view in distributed_student[:2])
            if mode == "within_image":
                expected = manual_within_loss(all_student, targets)
                gradient = torch.autograd.grad(expected, all_student)
                for actual, reference in zip(distributed_student, gradient):
                    torch.testing.assert_close(actual.grad / 2, reference[subset], atol=3e-5, rtol=3e-4)
                value = result["loss"].detach()
                dist.all_reduce(value)
                torch.testing.assert_close(value / 2, expected)
            else:
                expected = manual_cross_loss(all_student, targets)
                gradient = torch.autograd.grad(expected, all_student[:2])
                for actual, reference in zip(distributed_student[:2], gradient):
                    torch.testing.assert_close(actual.grad / 2, reference[subset], atol=3e-5, rtol=3e-4)

            # An empty rank participates in the required collectives. Within
            # ordering still works on the other rank's valid internal regions.
            empty_boxes = boxes[subset].clone()
            if rank == 1:
                invalidate_geometry(empty_boxes)
            empty_student = tuple(view.detach().requires_grad_() for view in distributed_student)
            skipped = RegionOrderingLoss(mode)(empty_student,
                                               tuple(view[subset] for view in targets), empty_boxes)
            assert skipped["query_count"].item() == (24 if mode == "within_image" else 0)
            skipped["loss"].backward()
            if rank == 1 or mode == "cross_image":
                assert all(view.grad.count_nonzero() == 0 for view in empty_student[:2])
            invalidate_geometry(empty_boxes)
            skipped = RegionOrderingLoss(mode)(empty_student,
                                               tuple(view[subset] for view in targets), empty_boxes)
            assert skipped["loss"].item() == 0
    finally:
        dist.destroy_process_group()


class RegionalOrderingTest(unittest.TestCase):
    def test_grouped_geometry_matches_original_fraction_masks(self):
        torch.manual_seed(517)
        crops = torch.rand(5, 12, 5)
        crops[..., :2] *= .6
        crops[..., 2:4] = (crops[..., :2] + .2 + crops[..., 2:4] * .4).clamp_max(1)
        crops[..., 4] = torch.randint(2, (5, 12)).float()
        pairs = list(combinations(range(12), 2))
        first, second = zip(*pairs)
        coordinates = torch.cat((torch.maximum(crops[:, first, :2], crops[:, second, :2]),
                                 torch.minimum(crops[:, first, 2:4], crops[:, second, 2:4])), -1)
        region_boxes = torch.cat((coordinates, torch.zeros_like(coordinates[..., :1])), -1)
        counts = [196, 196] + [36] * 10
        actual = _contained_patch_masks(crops, coordinates, counts)
        for view, patches in enumerate(counts):
            paired = torch.stack((crops[:, view, None].expand_as(region_boxes), region_boxes), 2)
            fractions, _, _ = intersection_patch_fractions(paired.reshape(-1, 2, 5), patches, 0)
            expected = (fractions[:, 0] >= 1).reshape(5, len(pairs), patches)
            torch.testing.assert_close(actual[view], expected)

    def test_tiled_pooling_matches_autograd_for_probabilities_and_mixed_precision_gradients(self):
        torch.manual_seed(156)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                logits = (torch.randn(4, 16, 13) * .05).to(dtype).requires_grad_()
                images = torch.tensor([0, 2, 3])
                weights = torch.randint(2, (3, 4, 16)).float()
                weights[1, 2] = 0  # Padded region, with zero contribution/gradient.
                probabilities = (logits.index_select(0, images).float() / .1).softmax(-1)
                expected = weights @ probabilities / weights.sum(-1, keepdim=True).clamp_min(1)
                actual = region_probability_mean(logits, images, weights, .1, 16 * 13 * 4)
                torch.testing.assert_close(actual, expected)
                incoming = torch.randn_like(expected)
                gradient, = torch.autograd.grad((expected * incoming).sum(), logits)
                reference, = torch.autograd.grad((actual * incoming).sum(), logits)
                torch.testing.assert_close(reference, gradient, atol=2e-5 if dtype == torch.float32 else 2e-3, rtol=2e-3)
                self.assertEqual(reference[1].count_nonzero(), 0)

    def test_cross_cosines_do_not_save_per_query_reference_vectors(self):
        torch.manual_seed(982)
        students = torch.randn(32, 64, requires_grad=True)
        teachers = torch.randn(32, 64, requires_grad=True)
        bank = F.normalize(torch.randn(48, 64), dim=-1)
        indices = torch.stack([torch.randperm(48)[:31] for _ in range(32)])
        shapes = []
        def save(value):
            shapes.append(tuple(value.shape))
            return value
        with torch.autograd.graph.saved_tensors_hooks(save, lambda value: value):
            student_cosines, teacher_cosines = _compare_queries(students, teachers, bank, indices)
        expected = torch.einsum("qd,qmd->qm", F.normalize(students, dim=-1), bank[indices])
        torch.testing.assert_close(student_cosines, expected)
        gradient, = torch.autograd.grad(expected.square().sum(), students)
        actual, = torch.autograd.grad(student_cosines.square().sum(), students)
        torch.testing.assert_close(actual, gradient, atol=2e-6, rtol=2e-5)
        self.assertNotIn((32, 31, 64), shapes)
        self.assertFalse(teacher_cosines.requires_grad)
        self.assertIsNone(teachers.grad)

    def test_teacher_sort_is_reused_and_never_recomputed_in_backward(self):
        student, teacher, targets, boxes = fixture(3)
        with mock.patch("losses.region_ordering_loss.bitonic_permutation", wraps=bitonic_permutation) as sorting:
            result = RegionOrderingLoss("within_image")(student, targets, boxes)
            calls = sorting.call_count
            self.assertLess(sum(call.args[0].shape[0] for call in sorting.call_args_list), result["query_count"].item())
            result["loss"].backward()
            self.assertEqual(sorting.call_count, calls)
        self.assertTrue(all(view.grad is None for view in teacher))

    def test_ragged_batched_pooling_and_query_groups_match_brute_force(self):
        student, _, targets, boxes = fixture(4)
        boxes[1, 2:, :4] = torch.tensor([0., 0., .5, .5])
        boxes[2, 2, :4] = torch.tensor([.01, .01, .02, .02])
        invalidate_geometry(boxes[3:])
        regions, masks = collect_overlap_regions(boxes, [64, 64, 16, 16, 16])
        families = [[], [], []]
        zero = sum(view.reshape(-1)[:1].sum() * 0 for view in student)
        for image, physical_regions in enumerate(regions):
            if len(physical_regions) < 3:
                continue
            reference_regions = torch.stack([
                targets[region.teacher_view][image, masks[region.teacher_view][image, region.row]].detach().mean(0)
                for region in physical_regions
            ])
            for index, region in enumerate(physical_regions):
                references = F.normalize(reference_regions[[other for other in range(len(physical_regions)) if other != index]], dim=-1)
                for (teacher_view, student_view), family in region.queries.items():
                    teacher_query = targets[teacher_view][image, masks[teacher_view][image, region.row]].detach().mean(0)
                    student_query = (student[student_view][image, masks[student_view][image, region.row]] / .1).softmax(-1).mean(0)
                    families[family].append(permutation_cross_entropy(
                        (F.normalize(student_query, dim=0) @ references.T)[None],
                        (F.normalize(teacher_query, dim=0) @ references.T)[None],
                    ).squeeze())
        expected = zero + sum(weight * torch.stack(family).mean()
                              for weight, family in zip((.5, .25, .25), families))
        gradient = torch.autograd.grad(expected, student)
        result = RegionOrderingLoss("within_image")(student, targets, boxes)
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        for actual, reference in zip(student, gradient):
            torch.testing.assert_close(actual.grad, reference, atol=3e-5, rtol=3e-4)

    @unittest.skipUnless(importlib.util.find_spec("diffsort"), "Optional upstream parity check")
    def test_sorter_values_permutations_and_gradients_match_neco_dependency(self):
        from diffsort import DiffSortNet
        for size in (2, 3, 4, 5, 7, 8, 17, 49, 65, 191):
            with self.subTest(size=size):
                torch.manual_seed(size)
                similarities = torch.randn(3, size, requires_grad=True)
                actual_values, actual = bitonic_permutation(similarities)
                values, expected = DiffSortNet("bitonic", size, steepness=100)(similarities)
                torch.testing.assert_close(actual_values, values, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                gradient, = torch.autograd.grad(actual.square().sum(), similarities, retain_graph=True)
                reference, = torch.autograd.grad(expected.square().sum(), similarities)
                torch.testing.assert_close(gradient, reference, atol=2e-3, rtol=2e-3)

    def test_permutation_axes_rank_average_and_teacher_detachment(self):
        student = torch.tensor([[0., .9, -.9]], requires_grad=True)
        teacher = torch.tensor([[.9, -.9, 0.]], requires_grad=True)
        _, permutation = bitonic_permutation(student)
        torch.testing.assert_close(permutation.sum(1), torch.ones(1, 3))
        torch.testing.assert_close(permutation.sum(2), torch.ones(1, 3))
        self.assertEqual(permutation.argmax(1).tolist(), [[2, 0, 1]])
        _, target = bitonic_permutation(teacher)
        expected = -(target.detach() * permutation.clamp_min(1e-12).log()).sum() / 3
        actual = permutation_cross_entropy(student, teacher).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)
        ties = torch.ones(2, 5, requires_grad=True)
        permutation_cross_entropy(ties, ties.detach()).sum().backward()
        self.assertTrue(torch.isfinite(ties.grad).all())

    def test_deduplication_flips_four_patch_threshold_and_best_teacher(self):
        _, _, _, boxes = fixture()
        regions, masks = collect_overlap_regions(boxes, [64, 64, 16, 16, 16])
        self.assertEqual([len(image) for image in regions], [6, 6])
        self.assertEqual([len(region.queries) for region in regions[0]], [2, 2, 2, 2, 2, 2])
        upper = regions[0][1]
        expected = torch.zeros(64, dtype=torch.bool)
        expected[cells(8, range(4), range(4, 8))] = True
        torch.testing.assert_close(masks[1][0, upper.row], expected)
        self.assertEqual(masks[0][0, regions[0][4].row].sum().item(), 4)
        self.assertEqual(regions[0][4].coordinates, (.25, .25, .5, .5))
        boxes[:, 1, :4] = torch.tensor([0, 0, .5, .5])
        regions, _ = collect_overlap_regions(boxes, [64, 64, 16, 16, 16])
        upper = next(region for region in regions[0] if region.coordinates == (0., 0., .5, .5))
        self.assertEqual(upper.teacher_view, 1)
        self.assertIn((0, 1), upper.queries)
        self.assertIn((1, 0), upper.queries)
        self.assertIn((1, 2), upper.queries)

    def test_teacher_coverage_filter_and_invalid_local_has_no_gradient(self):
        student, teacher, targets, boxes = fixture()
        tiny = torch.tensor([[[.01, .01, .02, .02, 0]]] * 2)
        boxes = torch.cat([boxes, tiny], 1)
        extra = torch.randn(2, 16, 7, requires_grad=True)
        result = RegionOrderingLoss("within_image")((*student, extra), targets, boxes)
        self.assertEqual(result["reference_region_count"].item(), 12)
        result["loss"].backward()
        self.assertEqual(extra.grad.count_nonzero(), 0)
        self.assertTrue(all(view.grad is None for view in teacher))
        # Local/local outside both global teachers is not a reference/query.
        boxes[:, :2, :4] = torch.tensor([0, 0, .25, .25])
        regions, _ = collect_overlap_regions(boxes, [64, 64, 16, 16, 16, 16])
        for image in regions:
            self.assertTrue(all(region.coordinates[2] <= .25 and region.coordinates[3] <= .25 for region in image))

    def test_within_loss_and_gradients_match_independently_selected_regions(self):
        student, teacher, targets, boxes = fixture()
        expected = manual_within_loss(student, targets)
        gradient = torch.autograd.grad(expected, student)
        result = RegionOrderingLoss("within_image")(student, targets, boxes)
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(result["query_count"].item(), 24)
        self.assertEqual(result["references_per_query"].item(), 5)
        result["loss"].backward()
        for actual, reference in zip(student, gradient):
            torch.testing.assert_close(actual.grad, reference, atol=3e-5, rtol=3e-4)
            self.assertGreater(actual.grad.abs().sum().item(), 0)
        self.assertTrue(all(view.grad is None for view in teacher))

    def test_cross_sampling_excludes_query_image_balances_images_and_does_not_retune(self):
        pools = {(0, 0): list(range(20)), (0, 1): list(range(20, 25)),
                 (1, 0): list(range(25, 35)), (1, 1): list(range(35, 37))}
        selected = sample_external_references(pools, (0, 0), 8, torch.Generator().manual_seed(0))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(all(index >= 20 for index in selected))
        self.assertEqual(sum(index >= 35 for index in selected), 2)
        self.assertEqual(sum(20 <= index < 25 for index in selected), 3)
        self.assertEqual(sum(25 <= index < 35 for index in selected), 3)
        self.assertIsNone(sample_external_references(pools, (0, 0), 18, torch.Generator()))
        student, _, targets, boxes = fixture(3)
        with mock.patch("losses.region_ordering_loss._compare_queries", wraps=_compare_queries) as compare:
            cross = RegionOrderingLoss("cross_image")(student, targets, boxes)
        indices = torch.cat([call.args[3] for call in compare.call_args_list])
        self.assertEqual(indices.shape, (6, 2))
        for image in range(3):
            self.assertTrue((indices[image * 2:image * 2 + 2] != image).all())
        self.assertTrue(all(len(set(row)) == 2 for row in indices.tolist()))
        self.assertEqual(cross["query_count"].item(), 6)
        self.assertEqual(cross["reference_region_count"].item(), 3)
        self.assertEqual(cross["global_local_queries"].item(), 0)
        self.assertEqual(cross["local_local_queries"].item(), 0)
        within = RegionOrderingLoss("within_image")(student, targets, boxes)
        self.assertEqual(within["query_count"].item(), 36)
        self.assertEqual(within["references_per_query"].item(), 5)

    def test_insufficient_reference_banks_and_independent_within_image_eligibility(self):
        for batch in (1, 2):
            student, _, targets, boxes = fixture(batch)
            result = RegionOrderingLoss("cross_image")(student, targets, boxes)
            self.assertEqual(result["query_count"].item(), 0)
            self.assertEqual(result["loss"].item(), 0)
            result["loss"].backward()
            self.assertTrue(all(view.grad.count_nonzero() == 0 for view in student[:2]))
            self.assertTrue(all(view.grad is None for view in student[2:]))
            within = RegionOrderingLoss("within_image")(student, targets, boxes)
            self.assertEqual(within["query_count"].item(), 12 * batch)
            boxes[:, 2:, :4] = torch.tensor([0., 0., .5, .5])
            result = RegionOrderingLoss("within_image")(student, targets, boxes)
            self.assertEqual(result["query_count"].item(), 0)
            result["loss"].backward()
            self.assertTrue(all(view.grad.count_nonzero() == 0 for view in student))

    def test_ibot_keeps_baseline_losses_and_adds_fixed_auxiliary(self):
        views, raw_teacher, _, boxes = fixture()
        cls = torch.randn(4, 3, requires_grad=True)
        local_cls = torch.randn(6, 3, requires_grad=True)
        output = (cls, torch.cat(views[:2]))
        teacher_output = (torch.randn(4, 3), torch.cat(raw_teacher))
        masks = [torch.rand(2, 8, 8) > .5 for _ in range(2)]
        baseline = make_loss(patch_out_dim=7, nlcrops=3, region_normalization="centering",
                             region_patch_threshold=1., lambda3=.4)
        baseline.center2.copy_(torch.linspace(-.02, .02, 7).reshape(1, 1, 7))
        targets = baseline.softmax_center_teacher(teacher_output, .07, .07)
        standard = baseline(output, targets, local_cls, masks, boxes[:, :2], teacher_patch_logits=teacher_output[1])
        for mode in ("within_image", "cross_image"):
            criterion = make_loss(patch_out_dim=7, nlcrops=3, region_normalization="centering",
                                  region_patch_threshold=1., lambda3=.4, loss_modality=mode)
            actual = criterion(output, targets, local_cls, masks, boxes, teacher_patch_logits=teacher_output[1],
                               student_local_patch_logits=torch.cat(views[2:]))
            for key in ("cls", "patch", "region", "region_raw"):
                torch.testing.assert_close(actual[key], standard[key])
            torch.testing.assert_close(actual["loss"], standard["loss"] + ORDERING_WEIGHT * actual["region_ordering_raw"])
            self.assertEqual(actual["region_weight"].item(), standard["region_weight"].item())
            criterion.lambda3 = 0
            with mock.patch.object(criterion.ordering_loss, "forward", side_effect=AssertionError("Must skip")):
                disabled = criterion(output, targets, local_cls, masks, None)
            self.assertEqual(disabled["region_ordering_active"].item(), 0)
        self.assertIsNone(baseline.ordering_loss)
        self.assertFalse(baseline.needs_local_patch_logits)

    def test_cross_uses_one_global_overlap_per_image_and_ignores_locals(self):
        student, teacher, targets, boxes = fixture(3)
        expected = manual_cross_loss(student, targets)
        gradient = torch.autograd.grad(expected, student[:2])
        result = RegionOrderingLoss("cross_image", min_area=.1)(student, targets, boxes)
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        for actual, reference in zip(student[:2], gradient):
            torch.testing.assert_close(actual.grad, reference, atol=3e-5, rtol=3e-4)
        self.assertTrue(all(view.grad is None for view in student[2:]))
        self.assertTrue(all(view.grad is None for view in teacher))
        modified = boxes.clone()
        modified[:, 2:, :4] = torch.tensor([.01, .01, .02, .02])
        globals_only = RegionOrderingLoss("cross_image", min_area=.1)(student[:2], targets, boxes[:, :2])
        changed_locals = RegionOrderingLoss("cross_image", min_area=.1)(student, targets, modified)
        torch.testing.assert_close(globals_only["loss"], result["loss"])
        torch.testing.assert_close(changed_locals["loss"], result["loss"])
        regions, masks = collect_overlap_regions(boxes[:, :2], [64, 64], global_only=True)
        self.assertEqual([len(image) for image in regions], [1, 1, 1])
        self.assertEqual([len(region.queries) for image in regions for region in image], [2, 2, 2])
        self.assertEqual(len(masks), 2)
        # Keep the standard global/global area filter for cross_image.
        modified[:, 1, :4] = torch.tensor([0., 0., .25, .25])
        filtered = RegionOrderingLoss("cross_image", min_area=.1)(student, targets, modified)
        self.assertEqual(filtered["reference_region_count"].item(), 0)
        self.assertEqual(filtered["loss"].item(), 0)
        criterion = make_loss(patch_out_dim=7, nlcrops=0, region_normalization="centering",
                              region_patch_threshold=1., loss_modality="cross_image")
        self.assertFalse(criterion.needs_local_patch_logits)
        output = (torch.randn(6, 3, requires_grad=True), torch.cat(student[:2]))
        teacher_output = (torch.randn(6, 3), torch.cat(teacher))
        normalized = criterion.softmax_center_teacher(teacher_output, .07, .07)
        losses = criterion(output, normalized, None, [torch.ones(3, 8, 8, dtype=torch.bool)] * 2,
                           boxes[:, :2], teacher_patch_logits=teacher_output[1])
        self.assertEqual(losses["region_ordering_query_count"].item(), 6)

    def test_configs_are_controlled_and_incompatible_modes_reject_early(self):
        base = Path("config/train.yaml").read_bytes()
        for mode in ("cross_image", "within_image"):
            path = Path(f"config/ablations/loss_modality_{mode}.yaml")
            expected = base.replace(b"loss_modality: standard", f"loss_modality: {mode}".encode())
            expected = expected.replace(b"batch_size_per_gpu: 64", b"batch_size_per_gpu: 48")
            self.assertEqual(path.read_bytes(), expected)
            config = load_config(path)
            self.assertEqual(config.batch_size_per_gpu, 48)
            self.assertEqual((config.loss_modality, config.lambda3, config.seed, config.additional_epochs), (mode, .4, 0, 50))
            self.assertEqual(config.initial_checkpoint, load_config(Path("config/train.yaml")).initial_checkpoint)
        for change in ({"loss_modality": "unknown"}, {"region_normalization": "deep"},
                       {"region_aggregation": "hellinger"}, {"region_patch_threshold": .5}, {"nlcrops": 0}):
            with self.assertRaisesRegex(ValueError, "loss_modality"):
                make_loss(**{"nlcrops": 3, "region_normalization": "centering",
                             "region_patch_threshold": 1., "loss_modality": "within_image", **change})
        for mode in ("cross_image", "within_image"):
            with self.assertRaisesRegex(ValueError, "loss_modality"):
                _validate_resume_compatibility({"args": SimpleNamespace()},
                                               SimpleNamespace(loss_modality=mode, lambda3=0))
        _validate_resume_compatibility({"args": SimpleNamespace()}, SimpleNamespace(loss_modality="standard", lambda3=0))

    def test_sampling_resume_and_checkpoint_centers_restore(self):
        student, _, targets, boxes = fixture(3)
        original = RegionOrderingLoss("cross_image", seed=19)
        original(student, targets, boxes)
        restored = RegionOrderingLoss("cross_image", seed=19)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        torch.testing.assert_close(original(student, targets, boxes)["loss"], restored(student, targets, boxes)["loss"])
        model = torch.nn.Linear(3, 3)
        teacher = torch.nn.Linear(3, 3)
        criterion = make_loss(nlcrops=3, region_normalization="centering", region_patch_threshold=1., loss_modality="cross_image")
        criterion.center.fill_(.3)
        criterion.center2.fill_(.4)
        criterion.ordering_loss.sampling_step.fill_(42)
        saved = {"student": model.state_dict(), "teacher": teacher.state_dict(), "ibot_loss": criterion.state_dict()}
        target = make_loss(nlcrops=3, region_normalization="centering", region_patch_threshold=1., loss_modality="cross_image")
        load_pretrained_state(saved, model, teacher, target)
        self.assertEqual(target.ordering_loss.sampling_step.item(), 42)
        torch.testing.assert_close(target.center, criterion.center)
        torch.testing.assert_close(target.center2, criterion.center2)
        previous_objective = copy.deepcopy(saved)
        del previous_objective["ibot_loss"]["ordering_loss.single_global_overlap"]
        with self.assertRaisesRegex(ValueError, "single-global-overlap"):
            load_resume_state(previous_objective, model, teacher, target, None, None)
        del saved["ibot_loss"]["ordering_loss.sampling_step"]
        with self.assertRaisesRegex(ValueError, "sampling_step"):
            load_resume_state(saved, model, teacher, target, None, None)

    def test_cpu_autocast_keeps_cosine_and_sorting_in_float32(self):
        student, _, targets, boxes = fixture()
        expected = RegionOrderingLoss("within_image")(student, targets, boxes)["loss"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = RegionOrderingLoss("within_image")(student, targets, boxes)
        self.assertEqual(result["loss"].dtype, torch.float32)
        torch.testing.assert_close(result["loss"], expected, atol=0, rtol=0)
        result["loss"].backward()
        self.assertTrue(all(torch.isfinite(view.grad).all() for view in student))

    def test_teacher_globals_only_student_masked_globals_and_ten_locals(self):
        def network(masked):
            return MultiCropWrapper(
                VisionTransformer(img_size=[224], patch_size=16, embed_dim=12,
                                  depth=1, num_heads=3, return_all_tokens=True,
                                  masked_im_modeling=masked, num_register_tokens=2),
                iBOTHead(12, 7, patch_out_dim=7, hidden_dim=16, bottleneck_dim=4,
                         nlayers=2, shared_head=False, norm_last_layer=False),
            )
        torch.manual_seed(531)
        student, teacher = network(True), network(False)
        teacher.load_state_dict(student.state_dict(), strict=False)
        teacher.requires_grad_(False)
        seen = {"teacher": [], "student": []}
        hooks = [model.backbone.register_forward_pre_hook(
            lambda module, args, name=name: seen[name].append(args[0].shape)
        ) for name, model in (("teacher", teacher), ("student", student))]
        try:
            images = [torch.randn(2, 3, 224, 224) for _ in range(2)]
            images += [torch.randn(2, 3, 96, 96) for _ in range(10)]
            masks = [torch.ones(2, 14, 14, dtype=torch.bool) for _ in range(2)]
            with torch.no_grad():
                teacher_output = teacher(images[:2])
            student_output = student(images[:2], mask=masks)
            student.backbone.masked_im_modeling = False
            local_cls, local_patch = student(images[2:])
            self.assertEqual(seen["teacher"], [torch.Size([4, 3, 224, 224])])
            self.assertEqual(seen["student"], [torch.Size([4, 3, 224, 224]), torch.Size([20, 3, 96, 96])])
            self.assertEqual(local_patch.shape, (20, 36, 7))
            criterion = make_loss(out_dim=7, patch_out_dim=7, nlcrops=10, lambda3=.4,
                                  region_normalization="centering", region_patch_threshold=1.,
                                  loss_modality="within_image")
            targets = criterion.softmax_center_teacher(teacher_output, .07, .07)
            _, _, _, source_boxes = fixture()
            boxes = torch.cat([source_boxes, source_boxes[:, 2:].repeat(1, 3, 1)[:, :7]], 1)
            result = criterion(student_output, targets, local_cls, masks, boxes,
                               teacher_patch_logits=teacher_output[1], student_local_patch_logits=local_patch)
            self.assertGreater(result["region_ordering_query_count"].item(), 0)
            result["region_ordering"].backward()
            self.assertGreater(student.head.last_layer2.weight_v.grad.abs().sum().item(), 0)
            self.assertGreater(student.backbone.patch_embed.proj.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        finally:
            for hook in hooks:
                hook.remove()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo required")
    def test_distributed_banks_ddp_query_averaging_and_empty_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=((Path(directory) / "rendezvous").as_uri(),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
