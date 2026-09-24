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
        path = Path(__file__).parents[1] / 'config' / 'train.yaml'
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

    def test_active_config_loads_without_lr_warmup(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        config = load_config(path)
        self.assertEqual(config.additional_epochs, values["additional_epochs"])
        self.assertEqual(config.epochs, values["additional_epochs"])
        self.assertEqual(config.precision, values["precision"])
        self.assertEqual(config.lambda3, values["lambda3"])
        self.assertEqual(config.region_temp, values["region_temp"])
        self.assertEqual(config.region_patch_threshold, values["region_patch_threshold"])
        self.assertEqual(config.region_normalization, values["region_normalization"])
        self.assertEqual(config.warmup_epochs, 0)
        self.assertIsNone(config.resume_checkpoint)

    def test_region_settings_load_and_validate(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        values.update(region_temp=.2, region_patch_threshold=.8)
        with mock.patch.object(Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))):
            config = load_config(path)
        self.assertEqual(config.region_temp, .2)
        self.assertEqual(config.region_patch_threshold, .8)
        for key, value in (("region_temp", 0), ("region_patch_threshold", 1.1),
                           ("region_min_area", 1.1)):
            invalid = dict(values, **{key: value})
            with mock.patch.object(Path, "open", mock.mock_open(read_data=yaml.safe_dump(invalid))):
                with self.assertRaisesRegex(ValueError, key):
                    load_config(path)

    def test_head_topology_is_selectable_from_yaml(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        for shared in (True, False):
            with self.subTest(shared=shared):
                configured = dict(values, shared_head=shared)
                with mock.patch.object(
                    Path, "open", mock.mock_open(read_data=yaml.safe_dump(configured))
                ):
                    config = load_config(path)
                self.assertEqual(config.shared_head, shared)

    def test_shared_region_normalization_selector(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        for mode in ("centering", "softmax", "raw_logits", "sinkhorn"):
            values["region_normalization"] = mode
            with mock.patch.object(Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))):
                self.assertEqual(load_config(path).region_normalization, mode)
        values["region_normalization"] = "invalid"
        with mock.patch.object(Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))):
            with self.assertRaisesRegex(ValueError, "region_normalization"):
                load_config(path)

    def test_head_topology_must_be_boolean(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        values["shared_head"] = "false"
        with mock.patch.object(
            Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))
        ):
            with self.assertRaisesRegex(ValueError, "shared_head must be a boolean"):
                load_config(path)

    def test_ibot_plus_plus_must_be_boolean(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        values["ibot_plus_plus"] = "true"
        with mock.patch.object(
            Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))
        ):
            with self.assertRaisesRegex(ValueError, "ibot_plus_plus must be a boolean"):
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
            config = load_config(Path(__file__).parents[1] / "config" / "train.yaml")

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

            with mock.patch.object(loss, "update_center", wraps=loss.update_center) as update:
                targets = get_teacher_targets(teacher, loss, epoch)
            update.assert_called_once()
            for actual, reference in zip(targets, expected):
                torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(
                loss.center, old_cls_center * 0.9 + teacher[0].mean(0) * 0.1
            )
            torch.testing.assert_close(
                loss.center2,
                old_patch_center * 0.9 + teacher[1].mean((0, 1)) * 0.1,
            )

    def test_teacher_targets_leave_raw_logits_unchanged(self):
        loss = self._loss()
        teacher = (torch.randn(4, 3, requires_grad=True), torch.randn(4, 4, 5, requires_grad=True))
        originals = tuple(x.detach().clone() for x in teacher)
        targets = get_teacher_targets(teacher, loss, 0)
        for raw, before, target in zip(teacher, originals, targets):
            torch.testing.assert_close(raw, before)
            self.assertFalse(target.requires_grad)


if __name__ == "__main__":
    unittest.main()
