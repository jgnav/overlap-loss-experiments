import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from losses import iBOTLoss
from train import get_teacher_targets, load_config
from utils import training as utils


class ContinuationConfigTest(unittest.TestCase):
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

            targets = get_teacher_targets(teacher, loss, epoch, "centering")

            for actual, reference in zip(targets, expected):
                torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(
                loss.center, old_cls_center * 0.9 + teacher[0].mean(0) * 0.1
            )
            torch.testing.assert_close(
                loss.center2,
                old_patch_center * 0.9 + teacher[1].mean((0, 1)) * 0.1,
            )

    def test_sk_uses_both_temperature_schedules_and_never_updates_centers(self):
        loss = self._loss()
        teacher = (torch.randn(4, 3) * 0.1, torch.randn(4, 4, 5) * 0.1)
        loss.center.fill_(0.1)
        loss.center2.fill_(-0.2)
        loss.update_center = mock.Mock(
            side_effect=AssertionError("SK must not update centers")
        )
        loss.softmax_center_teacher = mock.Mock(
            side_effect=AssertionError("SK must not use centering")
        )
        for epoch in (0, 1):
            expected = loss.sinkhorn_knopp_teacher(
                teacher,
                loss.teacher_temp_schedule[epoch],
                loss.teacher_temp2_schedule[epoch],
            )

            targets = get_teacher_targets(teacher, loss, epoch, "sinkhorn_knopp")

            for actual, reference in zip(targets, expected):
                torch.testing.assert_close(actual, reference)
        loss.update_center.assert_not_called()
        loss.softmax_center_teacher.assert_not_called()
        torch.testing.assert_close(loss.center, torch.full_like(loss.center, 0.1))
        torch.testing.assert_close(loss.center2, torch.full_like(loss.center2, -0.2))


if __name__ == "__main__":
    unittest.main()
