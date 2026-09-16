import datetime
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from model.head import iBOTHead
from model.vision_transformer import VisionTransformer
from utils.training import MultiCropWrapper
from utils.collapse_diagnostics import (
    FeatureCollapseDiagnostics, effective_prototype_weight,
    prototype_geometry_metrics, unique_prototype_counts,
)


def direct_feature_metrics(features):
    features = features.double()
    centered = features - features.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / len(features)).clamp_min(0)
    variance = eigenvalues.sum()
    p = eigenvalues / variance if variance > 0 else eigenvalues
    positive = p[p > 0]
    rank = (-(positive * positive.log()).sum()).exp().item() if variance > 0 else 0
    return {
        "teacher_patch_feature_effective_rank": rank,
        "teacher_patch_feature_effective_rank_ratio": rank / features.shape[1],
        "teacher_patch_feature_top1_variance_fraction": p[-1].item(),
        "teacher_patch_feature_mean_direction_norm": nn.functional.normalize(features, dim=1).mean(0).norm().item(),
        "teacher_patch_feature_sample_count": len(features),
    }


def distributed_diagnostics_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        cases = [
            (torch.tensor([[1., 0.], [-1., 0.]] * 2), torch.tensor([[0., 1.], [0., -1.]])),
            (torch.tensor([[1., 0.]] * 3), torch.tensor([[-1., 0.]])),
            (torch.empty(0, 2), torch.tensor([[1., 0.], [-1., 0.]])),
        ]
        for parts in cases:
            diagnostics = FeatureCollapseDiagnostics(2, 100)
            diagnostics.update(parts[rank].reshape(1, -1, 2))
            actual = diagnostics.compute()
            expected = direct_feature_metrics(torch.cat(parts))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-10, atol=1e-10)

        # Only rank zero reads prototype weights; identical scalar metrics reach both.
        model = SimpleNamespace(head=iBOTHead(3, 4, bottleneck_dim=2, nlayers=1, shared_head=True))
        with mock.patch.object(model.head, "prototype_layers", wraps=model.head.prototype_layers) as layers:
            metrics = prototype_geometry_metrics(model, model, chunk_size=2)
            assert layers.call_count == (2 if rank == 0 else 0)
        assert len(metrics) == 16
        gathered = [None, None]
        dist.all_gather_object(gathered, metrics)
        assert gathered[0] == gathered[1]
    finally:
        dist.destroy_process_group()


class PrototypeGeometryTest(unittest.TestCase):
    def test_duplicates_antipodes_and_chunk_invariance(self):
        vectors = torch.tensor([[1., 0.], [2., 0.], [-1., 0.], [0., 1.]])
        for chunk in (1, 2, 3, 256):
            self.assertEqual(unique_prototype_counts(vectors, chunk_size=chunk),
                             {0.025: 3, 0.1: 3, 0.25: 3, 0.5: 3})

    def test_representative_cover_does_not_merge_a_transitive_chain(self):
        angles = torch.tensor([0., 20., 40.], dtype=torch.float64) * math.pi / 180
        vectors = torch.stack((angles.cos(), angles.sin()), 1)
        self.assertEqual(unique_prototype_counts(vectors),
                         {0.025: 3, 0.1: 2, 0.25: 1, 0.5: 1})
        # Choosing the middle vector first gives another valid, smaller cover.
        self.assertEqual(unique_prototype_counts(vectors[[1, 0, 2]])[0.1], 1)

    def test_distance_threshold_is_strict(self):
        self.assertEqual(unique_prototype_counts(torch.eye(2), (1.0,)), {1.0: 2})

    def test_effective_weight_recomputes_current_g_and_v_without_mutation(self):
        layer = nn.utils.weight_norm(nn.Linear(2, 2, bias=False))
        cached = layer.weight.detach().clone()
        with torch.no_grad():
            layer.weight_v.copy_(torch.tensor([[2., 0.], [3., 0.]]))
            layer.weight_g.copy_(torch.tensor([[2.], [-3.]]))
        actual = effective_prototype_weight(layer)
        torch.testing.assert_close(actual, torch.tensor([[2., 0.], [-3., 0.]]))
        torch.testing.assert_close(layer.weight, cached)
        self.assertEqual(unique_prototype_counts(actual)[0.025], 2)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))

    def test_parametrized_weight_norm_is_supported(self):
        layer = nn.utils.parametrizations.weight_norm(nn.Linear(3, 4, bias=False))
        torch.testing.assert_close(effective_prototype_weight(layer), layer.weight)

    def test_shared_and_separate_patch_layers_follow_head_forward(self):
        for shared in (False, True):
            for bottleneck in (0, 3):
                head = iBOTHead(4, 5, patch_out_dim=7, hidden_dim=8,
                                bottleneck_dim=bottleneck, shared_head=shared)
                layers = head.prototype_layers()
                self.assertEqual(set(layers), {"patch"} if shared else {"patch", "cls"})
                patch = head.last_layer2 if bottleneck else head.mlp2
                self.assertIs(layers["patch"], patch)
                metrics = prototype_geometry_metrics(SimpleNamespace(head=head), SimpleNamespace(head=head), 2)
                self.assertEqual(len(metrics), 16 if shared else 32)
                self.assertIn("teacher_patch_unique_prototypes_eps_0025", metrics)
                for suffix in ("0025", "0100", "0250", "0500"):
                    prefix = "student_patch_unique_prototype"
                    count = metrics[f"{prefix}s_eps_{suffix}"]
                    ratio = metrics[f"{prefix}_ratio_eps_{suffix}"]
                    self.assertEqual(ratio, count / (5 if shared else 7))

    def test_invalid_prototype_vectors_are_explicit(self):
        for vectors in (torch.zeros(2, 3), torch.full((2, 3), float("nan"))):
            with self.assertRaisesRegex(ValueError, "finite, nonzero"):
                unique_prototype_counts(vectors)


class FeatureGeometryTest(unittest.TestCase):
    def test_isotropic_rank_one_and_identical_clouds(self):
        cases = [
            (torch.cat((torch.eye(3), -torch.eye(3))), 3., 1/3, 0.),
            (torch.tensor([[1., 0., 0.], [-1., 0., 0.]]), 1., 1., 0.),
            (torch.ones(5, 3), 0., 0., 1.),
            (torch.zeros(5, 3), 0., 0., 0.),
        ]
        for features, expected_rank, expected_top1, expected_direction in cases:
            diagnostics = FeatureCollapseDiagnostics(3)
            diagnostics.update(features.unsqueeze(0))
            result = diagnostics.compute()
            self.assertAlmostEqual(result["teacher_patch_feature_effective_rank"], expected_rank)
            self.assertAlmostEqual(result["teacher_patch_feature_effective_rank_ratio"], expected_rank / 3)
            self.assertAlmostEqual(result["teacher_patch_feature_top1_variance_fraction"], expected_top1)
            self.assertAlmostEqual(result["teacher_patch_feature_mean_direction_norm"], expected_direction)

    def test_sufficient_statistics_match_direct_covariance_over_multiple_batches(self):
        features = torch.randn(4, 9, 5, generator=torch.Generator().manual_seed(11)) + 4
        diagnostics = FeatureCollapseDiagnostics(5, 100)
        diagnostics.update(features[:2])
        diagnostics.update(features[2:])
        actual = diagnostics.compute()
        expected = direct_feature_metrics(features.flatten(0, 1))
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=10)

    def test_sampling_is_bounded_spans_both_views_and_does_not_touch_rng_or_features(self):
        features = torch.arange(2 * 6 * 3).reshape(2, 6, 3).float().requires_grad_()
        before = features.detach().clone()
        rng = torch.get_rng_state().clone()
        diagnostics = FeatureCollapseDiagnostics(3, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            diagnostics.update(features)
        sampled = features.detach().flatten(0, 1)[[0, 3, 6, 9]].double()
        torch.testing.assert_close(diagnostics.sum_features, sampled.sum(0))
        torch.testing.assert_close(diagnostics.sum_outer, sampled.T @ sampled)
        torch.testing.assert_close(features, before)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertFalse(diagnostics.sum_outer.requires_grad)
        self.assertEqual(diagnostics.compute()["teacher_patch_feature_sample_count"], 4)

    def test_backbone_feature_return_preserves_head_outputs_and_excludes_cls(self):
        backbone = VisionTransformer(img_size=[32], patch_size=16, embed_dim=12,
                                     depth=1, num_heads=3, return_all_tokens=True)
        model = MultiCropWrapper(backbone, iBOTHead(12, 5, nlayers=1, bottleneck_dim=3,
                                                   shared_head=True)).eval()
        images = [torch.randn(2, 3, 32, 32) for _ in range(2)]
        with torch.no_grad():
            normal = model(images)
            features, observed = model(images, return_backbone_feat=True)
            expected_features = backbone(torch.cat(images))
        for actual, expected in zip(observed, normal):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(features, expected_features)
        diagnostics = FeatureCollapseDiagnostics(12)
        diagnostics.update(features[:, 1:])
        self.assertEqual(diagnostics.compute()["teacher_patch_feature_sample_count"], 16)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_global_statistics_and_rank_zero_prototype_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_diagnostics_worker,
                     args=((Path(directory) / "rendezvous").as_uri(),), nprocs=2, join=True)


class CPUWrappedStudent(nn.Module):
    """Expose DDP's module attribute without initializing distributed training."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TrainingDiagnosticIntegrationTest(unittest.TestCase):
    def _run_epoch(self, sampled_batches):
        from train import train_one_epoch
        from losses import iBOTLoss

        torch.manual_seed(123)
        def make_model(masked):
            backbone = VisionTransformer(
                img_size=[32], patch_size=16, embed_dim=12, depth=1,
                num_heads=3, return_all_tokens=True, masked_im_modeling=masked,
            )
            return MultiCropWrapper(
                backbone, iBOTHead(12, 5, nlayers=1, bottleneck_dim=3, shared_head=True)
            )
        student = CPUWrappedStudent(make_model(True))
        teacher = make_model(False)
        teacher.load_state_dict(student.module.state_dict(), strict=False)
        teacher.requires_grad_(False)
        loss = iBOTLoss(
            out_dim=5, patch_out_dim=5, ngcrops=2, nlcrops=0,
            warmup_teacher_temp=.07, teacher_temp=.07,
            warmup_teacher_temp2=.07, teacher_temp2=.07,
            warmup_teacher_temp_epochs=0, nepochs=1, lambda3=.2,
        )
        optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
        batches = [(
            [torch.randn(2, 3, 32, 32) for _ in range(2)],
            torch.zeros(2),
            [torch.ones(2, 2, 2, dtype=torch.bool) for _ in range(2)],
            torch.tensor([[[0., 0., 1., 1., 0.], [.25, 0., 1., 1., 0.]]] * 2),
        ) for _ in range(2)]
        args = SimpleNamespace(
            source_checkpoint_epoch=800, epochs=1, print_freq=100,
            precision="fp32", global_crops_number=2, use_masked_im_modeling=True,
            clip_grad=0, freeze_last_layer=0,
            diagnostic_feature_batches=sampled_batches,
            diagnostic_max_patch_features_per_batch=8,
        )
        diagnostics = FeatureCollapseDiagnostics(12, 8)
        with (
            mock.patch.object(torch.Tensor, "cuda", lambda tensor, **kwargs: tensor),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("train.utils.concat_all_gather", side_effect=lambda tensor: tensor),
            mock.patch("train.FeatureCollapseDiagnostics", return_value=diagnostics),
            mock.patch.object(diagnostics, "update", wraps=diagnostics.update) as update,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            stats = train_one_epoch(
                student, teacher, teacher, loss, batches, optimizer,
                [.001] * 2, [.04] * 2, [.99] * 2, 0, None, args,
            )
        return stats, student.state_dict(), teacher.state_dict(), update.call_count

    def test_diagnostics_preserve_loss_and_optimizer_ema_updates_exactly(self):
        baseline, student_before, teacher_before, _ = self._run_epoch(0)
        observed, student_after, teacher_after, calls = self._run_epoch(1)
        self.assertEqual(calls, 1)
        self.assertEqual(observed["teacher_patch_feature_sample_count"], 8)
        self.assertNotIn("student_patch_effective_prototypes", observed)
        self.assertNotIn("teacher_patch_effective_prototypes", observed)
        for key in baseline:
            if not key.startswith("teacher_patch_feature_"):
                self.assertEqual(observed[key], baseline[key], key)
        for before, after in ((student_before, student_after), (teacher_before, teacher_after)):
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
