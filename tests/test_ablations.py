from pathlib import Path
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml

from losses.region_loss import RegionLoss
from tests.test_region_loss import boxes_full, boxes_disjoint
from tests.test_region_normalization import reference_sk
from train import load_config
from utils.checkpoint import _validate_resume_compatibility


ROOT = Path(__file__).resolve().parents[1]


class WeightedRegionTest(unittest.TestCase):
    def test_weighted_loss_and_gradients_match_arithmetic_reference_in_all_modes(self):
        torch.manual_seed(13)
        for mode in ('centering', 'softmax', 'raw_logits', 'sinkhorn'):
            for left, flipped, weights0 in (
                (.25, False, [.5, 1., .5, 1.]),
                (.25, True, [1., .5, 1., .5]),
                (.75, False, [0., .5, 0., .5]),
            ):
                with self.subTest(mode=mode, left=left, flipped=flipped):
                    student = (torch.randn(2, 2, 4, 3) * .1).requires_grad_()
                    teacher = (torch.randn(2, 2, 4, 3) * .1).requires_grad_()
                    boxes = boxes_disjoint(2)
                    boxes[0] = boxes_full()[0]
                    boxes[0, 1, 0] = left
                    boxes[0, 0, 4] = float(flipped)
                    weights = torch.tensor([weights0, [1.] * 4])
                    teacher_targets = ((teacher - .03) / .07).softmax(-1)
                    kwargs = {'teacher_patch_targets': tuple(teacher_targets.unbind(1))} if mode == 'centering' else {}
                    result = RegionLoss(patch_threshold='weighted', temperature=.2,
                                        normalization=mode, student_temperature=.1)(
                        tuple(student.unbind(1)), tuple(teacher.unbind(1)), boxes, **kwargs)
                    torch.testing.assert_close(result['patch_weights'][0], weights)
                    self.assertEqual(result['patch_weights'][1].count_nonzero(), 0)
                    regions = []
                    for index, logits in enumerate((student[0], teacher[0].detach())):
                        if mode == 'raw_logits':
                            probabilities = F.normalize(logits, dim=-1)
                        elif mode == 'sinkhorn':
                            probabilities = torch.zeros_like(logits)
                            probabilities[weights > 0] = reference_sk(logits[weights > 0], .2)
                        elif mode == 'centering':
                            probabilities = (logits / .1).softmax(-1) if index == 0 else teacher_targets[0].detach()
                        else:
                            probabilities = (logits / .2).softmax(-1)
                        regions.append((probabilities * weights[..., None]).sum(1) / weights.sum(1, keepdim=True))
                    s, t = regions
                    if mode == 'raw_logits':
                        expected = 1 - .5 * (F.cosine_similarity(t[0], s[1], dim=0)
                                               + F.cosine_similarity(t[1], s[0], dim=0))
                    else:
                        expected = -.5 * ((t[0] * s[1].log()).sum() + (t[1] * s[0].log()).sum())
                    gradient, = torch.autograd.grad(expected, student)
                    result['loss'].backward()
                    torch.testing.assert_close(result['loss'], expected)
                    torch.testing.assert_close(student.grad, gradient, atol=2e-6, rtol=2e-5)
                    self.assertTrue(torch.isfinite(student.grad).all())
                    self.assertEqual(student.grad[1].count_nonzero(), 0)
                    self.assertIsNone(teacher.grad)

    def test_weighted_full_overlap_equals_thresholded_and_no_overlap_has_zero_gradient(self):
        torch.manual_seed(19)
        student = tuple(torch.randn(1, 4, 3, requires_grad=True) for _ in range(2))
        teacher = tuple(torch.randn(1, 4, 3) for _ in range(2))
        weighted = RegionLoss(patch_threshold='weighted')
        torch.testing.assert_close(weighted(student, teacher, boxes_full())['loss'],
                                   RegionLoss()(student, teacher, boxes_full())['loss'])
        result = weighted(student, teacher, boxes_disjoint())
        self.assertEqual(result['loss'].item(), 0)
        result['loss'].backward()
        self.assertTrue(all(x.grad.count_nonzero() == 0 for x in student))
        self.assertEqual(weighted(student, teacher, boxes_full())['valid_ratio'].item(), 1)
        filtered = RegionLoss(min_area=.8, patch_threshold='weighted')
        boxes = boxes_full()
        boxes[0, 1, 0] = .25
        self.assertEqual(filtered(student, teacher, boxes)['loss'].item(), 0)

    def test_weighted_mode_cannot_silently_resume_thresholded_checkpoint(self):
        saved = {'args': SimpleNamespace(region_patch_threshold=.5)}
        with self.assertRaisesRegex(ValueError, 'region_patch_threshold'):
            _validate_resume_compatibility(saved, SimpleNamespace(region_patch_threshold='weighted', lambda3=0))


class AblationConfigTest(unittest.TestCase):
    def test_twelve_configs_change_only_their_factor_and_common_run_settings(self):
        base = yaml.safe_load((ROOT / 'config/train.yaml').read_text())
        base.update(additional_epochs=50, online_probes_enabled=True, output_dir='output/ablation')
        groups = {'region_min_area': ['0.10', '0.20', '0.30', '0.50'],
                  'lambda3': ['0.20', '0.50', '1.0', '2.0'],
                  'region_patch_threshold': ['0.2', '0.5', '0.8', 'weighted']}
        paths = list((ROOT / 'config/ablations').glob('*.yaml'))
        self.assertEqual(len(paths), 12)
        for key, values in groups.items():
            for value in values:
                path = ROOT / 'config/ablations' / (key + '_' + value.replace('.', 'p') + '.yaml')
                expected = dict(base, **{key: yaml.safe_load(value)})
                self.assertEqual(yaml.safe_load(path.read_text()), expected)
                args = load_config(path)
                self.assertEqual(args.epochs, 50)
                self.assertTrue(args.online_probes_enabled)
                self.assertEqual(args.gpu_count, 4)

    def test_launcher_is_a_twelve_task_array_with_identical_resources(self):
        script = (ROOT / 'slurm/ablation.sh').read_text()
        for setting in ('--array=0-11', '--gpus-per-node=4', '--cpus-per-task=24',
                        '--mem=128G', '--time=50:00:00',
                        '--partition=3090_risk,a100,rtx_pro6000_risk',
                        '--output=logs/abaltion/%A_%a.out'):
            self.assertIn('#SBATCH ' + setting, script)
        for path in (ROOT / 'config/ablations').glob('*.yaml'):
            self.assertIn(path.stem, script)


if __name__ == '__main__':
    unittest.main()
