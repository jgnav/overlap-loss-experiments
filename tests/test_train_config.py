import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from losses import iBOTLoss
from train import get_teacher_targets, load_config
from utils import training as utils


class ContinuationConfigTest(unittest.TestCase):
    def test_wandb_settings_come_from_yaml_instead_of_slurm_environment(self):
        path = Path(__file__).parents[1] / 'train.yaml'
        values = yaml.safe_load(path.read_text())
        values.update(wandb_mode='offline', wandb_run_id='yaml-id', wandb_resume='allow')
        with mock.patch.dict('os.environ', {'WANDB_MODE': 'online', 'WANDB_RUN_ID': 'env-id', 'WANDB_RESUME': 'must'}):
            with mock.patch.object(Path, 'open', mock.mock_open(read_data=yaml.safe_dump(values))):
                args = load_config(path)
            from train import assign_run_output_directory
            assign_run_output_directory(args)
        self.assertEqual(args.wandb_mode, 'offline')
        self.assertEqual(args.wandb_run_id, 'yaml-id')
        self.assertEqual(args.wandb_resume, 'allow')

    def test_production_config_is_bf16_200_epochs_without_lr_warmup(self):
        config = load_config(Path(__file__).parents[1] / "train.yaml")

        self.assertEqual(config.additional_epochs, 200)
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.precision, "bf16")
        self.assertFalse(config.use_fp16)
        self.assertEqual(config.warmup_epochs, 0)
        self.assertEqual(config.batch_size_per_gpu * config.gpu_count, 256)
        self.assertEqual(config.saveckp_freq, 50)
        self.assertEqual(config.centering, "centering")
        self.assertEqual(config.teacher_target_version, 2)

    def test_teacher_normalization_choices_and_legacy_default(self):
        path = Path(__file__).parents[1] / "train.yaml"
        config_values = yaml.safe_load(path.read_text())
        for mode in (None, "centering", "sinkhorn_knopp"):
            with self.subTest(mode=mode):
                values = dict(config_values)
                if mode is None:
                    values.pop("centering")
                else:
                    values["centering"] = mode
                with mock.patch.object(
                    Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))
                ):
                    config = load_config(path)
                self.assertEqual(config.centering, mode or "centering")

    def test_invalid_teacher_normalization_fails_at_config_load(self):
        path = Path(__file__).parents[1] / "train.yaml"
        values = yaml.safe_load(path.read_text())
        values["centering"] = "sinkhorn"
        with mock.patch.object(
            Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))
        ):
            with self.assertRaisesRegex(ValueError, "centering, sinkhorn_knopp"):
                load_config(path)

    def test_continuation_cosine_schedule_starts_at_configured_lr(self):
        schedule = utils.cosine_scheduler(
            1e-4,
            1e-6,
            epochs=200,
            iterations_per_epoch=2,
            warmup_epochs=0,
        )

        self.assertAlmostEqual(schedule[0], 1e-4)

    def test_debug_environment_can_override_precision_and_batch(self):
        with mock.patch.dict(
            "os.environ",
            {
                "IBOT_PRECISION_OVERRIDE": "fp32",
                "IBOT_BATCH_SIZE_PER_GPU_OVERRIDE": "4",
                "IBOT_GPU_COUNT_OVERRIDE": "1",
            },
        ):
            config = load_config(Path(__file__).parents[1] / "train.yaml")

        self.assertEqual(config.precision, "fp32")
        self.assertEqual(config.batch_size_per_gpu, 4)
        self.assertEqual(config.gpu_count, 1)


class TeacherTargetTrainingTest(unittest.TestCase):
    @staticmethod
    def _loss():
        return iBOTLoss(
            out_dim=3,
            patch_out_dim=5,
            ngcrops=2,
            nlcrops=0,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp2=0.05,
            teacher_temp2=0.09,
            warmup_teacher_temp_epochs=2,
            nepochs=3,
        )

    def test_centering_uses_previous_centers_then_updates_once_per_batch(self):
        loss = self._loss()
        teacher = (torch.randn(4, 3) * 0.1, torch.randn(4, 4, 5) * 0.1)
        for epoch in (0, 1):
            old_cls_center = loss.center.clone()
            old_patch_center = loss.center2.clone()
            expected = (
                ((teacher[0] - old_cls_center)
                 / loss.teacher_temp_schedule[epoch]).softmax(-1),
                ((teacher[1] - old_patch_center)
                 / loss.teacher_temp2_schedule[epoch]).softmax(-1),
            )

            targets, overlap_targets = get_teacher_targets(
                teacher, loss, epoch, "centering"
            )

            self.assertIsNone(overlap_targets)
            for actual, reference in zip(targets, expected):
                torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(
                loss.center, old_cls_center * 0.9 + teacher[0].mean(0) * 0.1
            )
            torch.testing.assert_close(
                loss.center2,
                old_patch_center * 0.9 + teacher[1].mean((0, 1)) * 0.1,
            )

    def test_sk_keeps_centered_targets_and_center_updates_in_both_schedules(self):
        centered_loss = self._loss()
        sk_loss = self._loss()
        teacher = (torch.randn(4, 3) * 0.1, torch.randn(4, 4, 5) * 0.1)
        boxes = torch.tensor(
            [[[0., 0., 1., 1., 0.], [0.25, 0., 1., 1., 0.]]] * 2
        )
        for epoch in (0, 1):
            centered, _ = get_teacher_targets(
                teacher, centered_loss, epoch, "centering", boxes
            )
            with mock.patch.object(
                sk_loss.region_loss, "sinkhorn_knopp_teacher",
                wraps=sk_loss.region_loss.sinkhorn_knopp_teacher,
            ) as build_overlap:
                sk, overlap = get_teacher_targets(
                    teacher, sk_loss, epoch, "sinkhorn_knopp", boxes
                )

            for actual, reference in zip(sk, centered):
                torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(sk_loss.center, centered_loss.center)
            torch.testing.assert_close(sk_loss.center2, centered_loss.center2)
            self.assertIsNotNone(overlap)
            build_overlap.assert_called_once()
            self.assertIs(build_overlap.call_args.args[0], teacher[1])
            self.assertEqual(
                build_overlap.call_args.args[2], sk_loss.teacher_temp2_schedule[epoch]
            )

    def test_sk_overlap_targets_are_independent_of_centers(self):
        teacher = (torch.randn(4, 3) * 0.1, torch.randn(4, 4, 5) * 0.1)
        boxes = torch.tensor(
            [[[0., 0., 1., 1., 0.], [0.5, 0., 1., 1., 0.]]] * 2
        )
        first = self._loss()
        second = self._loss()
        second.center.copy_(torch.tensor([[0.1, -0.1, 0.2]]))
        second.center2.copy_(torch.linspace(-0.3, 0.3, 5).reshape(1, 1, 5))

        targets_a, overlap_a = get_teacher_targets(
            teacher, first, 0, "sinkhorn_knopp", boxes
        )
        targets_b, overlap_b = get_teacher_targets(
            teacher, second, 0, "sinkhorn_knopp", boxes
        )

        torch.testing.assert_close(overlap_a.probabilities, overlap_b.probabilities)
        self.assertFalse(torch.allclose(targets_a[0], targets_b[0]))
        self.assertFalse(torch.allclose(targets_a[1], targets_b[1]))

    def test_zero_overlap_weight_skips_sk_and_matches_pure_ibot(self):
        centered_loss = self._loss()
        centered_loss.lambda3 = 0
        sk_loss = self._loss()
        sk_loss.lambda3 = 0
        sk_loss.region_loss.sinkhorn_knopp_teacher = mock.Mock(
            side_effect=AssertionError("No SK when lambda3 is zero")
        )
        teacher = (torch.randn(4, 3) * 0.1, torch.randn(4, 4, 5) * 0.1)
        student = (torch.randn(4, 3), torch.randn(4, 4, 5))
        masks = [torch.ones(2, 2, 2, dtype=torch.bool) for _ in range(2)]
        centered, _ = get_teacher_targets(teacher, centered_loss, 0, "centering")
        sk, overlap = get_teacher_targets(teacher, sk_loss, 0, "sinkhorn_knopp")

        expected = centered_loss(student, centered, None, masks, None)
        actual = sk_loss(student, sk, None, masks, None, teacher_overlap_targets=overlap)

        self.assertIsNone(overlap)
        sk_loss.region_loss.sinkhorn_knopp_teacher.assert_not_called()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
