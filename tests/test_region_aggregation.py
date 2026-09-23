import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from losses.region_aggregation import METHODS, RegionAggregation
from losses.region_loss import RegionLoss
from tests.test_region_loss import boxes_full, boxes_disjoint
from utils.checkpoint import _validate_resume_compatibility


class RegionAggregationTest(unittest.TestCase):
    def test_hellinger_formula_weights_and_gradients(self):
        loss = RegionLoss(aggregation='hellinger', temperature=1.)
        z = torch.tensor([[[2., -1., 0.], [-1., 3., 1.], [999., -999., 0.]]], requires_grad=True)
        w = torch.tensor([[1., .25, 0.]])
        actual = loss._region_log_distribution(z, w)
        p = z.softmax(-1)
        expected = (p[:, :2].sqrt() * w[:, :2, None]).sum(1).square()
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(actual.exp(), expected)
        torch.testing.assert_close(loss._region_probability_mean(p.detach(), w), expected.detach())
        ga = torch.autograd.grad(actual.sum(), z, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.log().sum(), z)[0]
        torch.testing.assert_close(ga, ge)
        self.assertEqual(ga[:, 2].count_nonzero(), 0)
        torch.testing.assert_close(actual, loss._region_log_distribution(z[:, [1, 0, 2]], w[:, [1, 0, 2]]))
        extreme = torch.tensor([[[1000., -1000.], [900., -900.]]], requires_grad=True)
        log_q = loss._region_log_distribution(extreme, torch.ones(1, 2))
        log_q.sum().backward()
        self.assertTrue(torch.isfinite(log_q).all())
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_hellinger_ablation_only_changes_method(self):
        root = Path(__file__).parents[1] / 'config'
        expected = (root / 'train.yaml').read_bytes().replace(
            b'region_aggregation: mean', b'region_aggregation: hellinger')
        self.assertEqual((root / 'ablations/region_aggregation_hellinger.yaml').read_bytes(), expected)

    def test_yaml_selector_and_resume_guard(self):
        from train import load_config
        path = Path(__file__).parents[1] / 'config/train.yaml'
        config = yaml.safe_load(path.read_text())
        for method in METHODS:
            values = dict(config, region_aggregation=method)
            with mock.patch.object(Path, 'open', mock.mock_open(read_data=yaml.safe_dump(values))):
                self.assertEqual(load_config(path).region_aggregation, method)
        args = SimpleNamespace(region_aggregation='mean')
        _validate_resume_compatibility({'args': {}}, args)
        args.region_aggregation = 'mean_variance'
        with self.assertRaisesRegex(ValueError, 'region_aggregation'):
            _validate_resume_compatibility({'args': {}}, args)
        _validate_resume_compatibility({'args': {'region_aggregation': 'mean_variance'}}, args)

    def test_projected_formulas_and_stopped_scale_gradient(self):
        torch.manual_seed(3)
        s = torch.randn(1, 4, 5).softmax(-1).requires_grad_()
        t = torch.randn(1, 4, 5).softmax(-1)
        weights = torch.ones(1, 4)
        for method in ('mean_projected_variance', 'mean_projected_covariance',
                       'mean_centered_swd', 'mean_normalized_swd', 'swd'):
            agg = RegionAggregation(method)
            actual = agg(s, t, weights, weights, torch.ones(1))
            w = agg.directions
            sp = (s - s.mean(1, keepdim=True)) @ w
            tp = (t - t.mean(1, keepdim=True)) @ w
            ss = (sp.square().mean(1) + 1e-8).sqrt()
            ts = (tp.square().mean(1) + 1e-8).sqrt()
            if method == 'mean_projected_variance':
                expected = 1 + (ss - ts).square().mean(-1)
            elif method == 'mean_projected_covariance':
                expected = 1 + ((sp.transpose(1, 2) @ sp - tp.transpose(1, 2) @ tp) / 4).square().sum((1, 2))
            else:
                base = torch.zeros(1) if method == 'swd' else torch.ones(1)
                if method == 'swd':
                    sp, tp = s @ w, t @ w
                elif method == 'mean_normalized_swd':
                    base = base + (ss - ts).square().mean(-1)
                    sp, tp = sp / (ss.detach()[:, None] + 1e-8), tp / (ts[:, None] + 1e-8)
                # Four equal-mass patches: midpoint quantiles repeat each
                # sorted value equally often, so sorted MSE is exact here.
                expected = base + (sp.sort(1).values - tp.sort(1).values).square().mean((1, 2))
            torch.testing.assert_close(actual, expected)
            ga = torch.autograd.grad(actual.sum(), s, retain_graph=True)[0]
            ge = torch.autograd.grad(expected.sum(), s, retain_graph=True)[0]
            torch.testing.assert_close(ga, ge, atol=2e-6, rtol=2e-5)

    def test_weighted_moments_match_direct_covariance(self):
        torch.manual_seed(4)
        s, t = torch.randn(2, 4, 7), torch.randn(2, 5, 7)
        sw = torch.tensor([[1., .2, 0., .8]]).expand(2, -1)
        tw = torch.tensor([[.1, 1., .3, .4, 0.]]).expand(2, -1)
        sm, sr, sw = RegionAggregation.moments(s, sw)
        tm, tr, tw = RegionAggregation.moments(t, tw)
        sc = sr.transpose(1, 2) @ (sr * sw[..., None])
        tc = tr.transpose(1, 2) @ (tr * tw[..., None])
        expected_cov = (sc - tc).square().sum((1, 2))
        torch.testing.assert_close(RegionAggregation.covariance_distance(sr, tr, sw, tw), expected_cov)
        sv, tv = sc.diagonal(dim1=1, dim2=2), tc.diagonal(dim1=1, dim2=2)
        for method, expected in (
            ('mean_covariance', expected_cov),
            ('mean_scalar_variance', ((sv.sum(-1) + 1e-8).sqrt() - (tv.sum(-1) + 1e-8).sqrt()).square()),
            ('mean_variance', ((sv + 1e-8).sqrt() - (tv + 1e-8).sqrt()).square().mean(-1)),
        ):
            torch.testing.assert_close(RegionAggregation(method)(s, t, sw, tw, torch.zeros(2)), expected)

    def test_weighted_quantiles_ignore_zero_mass_and_match_repeated_samples(self):
        agg = RegionAggregation('swd')
        x = torch.tensor([[[0.], [2.], [999.]]], requires_grad=True)
        w = torch.tensor([[.25, .75, 0.]])
        repeated = torch.tensor([[[0.], [2.], [2.], [2.]]])
        q = agg.empirical_quantiles(x, w)
        torch.testing.assert_close(q[..., :32], torch.zeros(1, 1, 32))
        torch.testing.assert_close(q[..., 32:], torch.full((1, 1, 96), 2.))
        torch.testing.assert_close(q, agg.empirical_quantiles(repeated, torch.ones(1, 4)))
        agg.swd(x, repeated + 1, w, torch.ones(1, 4)).sum().backward()
        self.assertEqual(x.grad[0, 2].item(), 0)

    def test_all_methods_permutation_invariant_and_teacher_detached(self):
        torch.manual_seed(8)
        for method in METHODS:
            s = torch.randn(2, 4, 6).softmax(-1).requires_grad_()
            t = torch.randn(2, 4, 6).softmax(-1).requires_grad_()
            w = torch.tensor([[.1, 1., 0., .4]]).expand(2, -1)
            agg = RegionAggregation(method)
            result = agg(s, t, w, w, s.sum((1, 2)) * 0)
            perm = torch.tensor([3, 2, 0, 1])
            shuffled = agg(s[:, perm], t, w[:, perm], w, s.sum((1, 2)) * 0)
            torch.testing.assert_close(result, shuffled, atol=1e-7, rtol=1e-5)
            result.sum().backward()
            self.assertIsNone(t.grad)
            self.assertTrue(torch.isfinite(s.grad).all(), method)
            self.assertEqual(s.grad[:, 2].count_nonzero(), 0)

    def test_all_methods_normalizations_and_empty_regions(self):
        torch.manual_seed(9)
        for method in METHODS:
            for mode in ('softmax', 'centering', 'sinkhorn', 'raw_logits'):
                if method == 'hellinger' and mode == 'raw_logits':
                    with self.assertRaisesRegex(ValueError, 'probability distributions'):
                        RegionLoss(normalization=mode, aggregation=method)
                    continue
                for threshold in (.75, 'weighted'):
                    s = tuple(torch.randn(2, 4, 5, requires_grad=True) for _ in range(2))
                    t = tuple(torch.randn(2, 4, 5, requires_grad=True) for _ in range(2))
                    boxes = boxes_full(2)
                    boxes[0, 1, 0] = .3
                    boxes[1:] = boxes_disjoint()
                    loss = RegionLoss(normalization=mode, aggregation=method, patch_threshold=threshold)
                    kw = dict(teacher_patch_targets=tuple((x.detach() / .07).softmax(-1) for x in t))
                    result = loss(s, t, boxes, **kw)
                    result['loss'].backward()
                    for v, x in enumerate(s):
                        self.assertTrue(torch.isfinite(x.grad).all(), (method, mode))
                        self.assertEqual(x.grad[~result['patch_mask'][:, v]].count_nonzero(), 0)
                    self.assertTrue(all(x.grad is None for x in t))
                    empty = loss(s, t, boxes_disjoint(2), **kw)['loss']
                    self.assertEqual(empty.item(), 0)
                    empty.backward()

    def test_projection_directions_are_reproducible_and_do_not_consume_rng(self):
        a, b = RegionAggregation('mean_projected_variance'), RegionAggregation('mean_projected_variance')
        state = torch.random.get_rng_state()
        x = torch.ones(2, 4, 7)
        torch.testing.assert_close(a.project(x), b.project(x))
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        torch.testing.assert_close(a.directions.norm(dim=0), torch.ones(64))


if __name__ == '__main__':
    unittest.main()
