import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.region_loss import RegionLoss
from model.head import iBOTHead
from model.vision_transformer import VisionTransformer
from tests.test_region_loss import boxes_full
from utils.training import MultiCropWrapper


class DeepRegionTest(unittest.TestCase):
    def test_raw_deep_averages_four_independent_cosine_losses(self):
        torch.manual_seed(41)
        loss = RegionLoss(patch_threshold=0.5, normalization="raw_logits_deep")
        boxes = boxes_full(2)
        boxes[0, 1, 0] = 0.25
        geometry = loss.prepare_geometry(boxes, 4)
        student = {depth: torch.randn(4, 4, 6, requires_grad=True)
                   for depth in (3, 6, 9, 12)}
        teacher = {depth: torch.randn(4, 4, 6, requires_grad=True)
                   for depth in (3, 6, 9, 12)}
        actual = loss.forward_deep(student, teacher, geometry)["loss"]
        weights, valid = geometry["weights"], geometry["valid"]
        expected_layers = []
        for depth in (3, 6, 9, 12):
            s0, s1 = student[depth].chunk(2)
            t0, t1 = teacher[depth].detach().chunk(2)
            def pool(features, view):
                mean = (features * weights[:, view, :, None]).sum(1)
                mean = mean / weights[:, view].sum(1, keepdim=True).clamp_min(1)
                return F.normalize(mean, dim=-1)
            s0, s1 = pool(s0, 0), pool(s1, 1)
            t0, t1 = pool(t0, 0), pool(t1, 1)
            expected_layers.append((1 - 0.5 * (
                F.cosine_similarity(t0, s1, dim=-1)
                + F.cosine_similarity(t1, s0, dim=-1)
            ))[valid].mean())
        expected = torch.stack(expected_layers).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for depth in student:
            self.assertIsNotNone(student[depth].grad)
            self.assertIsNone(teacher[depth].grad)
        self.assertEqual(geometry["selected"].shape, (2, 2, 4))

    def test_softmax_deep_keeps_layers_separate(self):
        torch.manual_seed(42)
        loss = RegionLoss(patch_threshold=0.5, normalization="softmax_deep")
        geometry = loss.prepare_geometry(boxes_full(2), 4)
        student = {depth: torch.randn(4, 7, requires_grad=True).log_softmax(-1)
                   for depth in (3, 6, 9, 12)}
        teacher = {depth: torch.randn(4, 7, requires_grad=True).log_softmax(-1)
                   for depth in (3, 6, 9, 12)}
        actual = loss.forward_deep(student, teacher, geometry)["loss"]
        terms = []
        for depth in student:
            s0, s1 = student[depth].chunk(2)
            t0, t1 = teacher[depth].detach().chunk(2)
            terms.append(-0.5 * ((t0.exp() * s1).sum(-1)
                                 + (t1.exp() * s0).sum(-1)).mean())
        torch.testing.assert_close(actual, torch.stack(terms).mean())
        actual.backward()
        for depth in student:
            self.assertIsNotNone(student[depth].grad_fn)
            self.assertIsNone(teacher[depth].grad)

    def test_streamed_softmax_matches_direct_patch_mean_and_gradient(self):
        torch.manual_seed(43)
        patches = torch.randn(2, 4, 3, requires_grad=True)
        projection = nn.Linear(3, 5)
        weights = torch.tensor([[1., 0., 0., 1.], [0., 0., 1., 0.]])
        actual = MultiCropWrapper._pool_patch_softmax(
            patches, weights, projection, 0.7, chunk_size=2
        )
        expected = ((projection(patches) / 0.7).softmax(-1)
                    * weights[..., None]).sum(1) / weights.sum(1, keepdim=True)
        torch.testing.assert_close(actual.exp(), expected)
        actual.sum().backward()
        first_grad = patches.grad.detach().clone()
        patches.grad = None
        expected.log().sum().backward()
        torch.testing.assert_close(patches.grad, first_grad)
        self.assertTrue(torch.isfinite(first_grad).all())

    def test_wrapper_uses_four_depths_and_excludes_registers(self):
        torch.manual_seed(44)
        backbone = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=12, depth=12,
            num_heads=3, return_all_tokens=True, num_register_tokens=4,
        )
        head = iBOTHead(12, 7, patch_out_dim=7, nlayers=1,
                        bottleneck_dim=0, shared_head=True)
        wrapped = MultiCropWrapper(backbone, head)
        views = [torch.randn(2, 3, 32, 32) for _ in range(2)]
        (_, patch_logits), deep = wrapped(views, return_deep_patches=True)
        self.assertEqual(tuple(deep), (3, 6, 9, 12))
        for features in deep.values():
            self.assertEqual(features.shape, (4, 4, 12))
        self.assertEqual(patch_logits.shape, (4, 4, 7))
        weights = (torch.ones(2, 4), torch.ones(2, 4))
        (_, _), pooled = wrapped(
            views, return_deep_patches=True,
            deep_region_weights=weights, deep_softmax_temperature=0.7,
        )
        for region in pooled.values():
            self.assertEqual(region.shape, (4, 7))
            torch.testing.assert_close(region.exp().sum(-1), torch.ones(4))
        expected_final = (patch_logits / 0.7).softmax(-1).mean(1)
        torch.testing.assert_close(pooled[12].exp(), expected_final)

    def test_deep_modes_require_binary_mean_pooling(self):
        for mode in ("raw_logits_deep", "softmax_deep"):
            with self.assertRaisesRegex(ValueError, "numeric patch threshold"):
                RegionLoss(normalization=mode, patch_threshold="weighted")
            with self.assertRaisesRegex(ValueError, "mean aggregation"):
                RegionLoss(normalization=mode, aggregation="mean_variance")


if __name__ == "__main__":
    unittest.main()
