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

    def test_production_config_is_bf16_200_epochs_without_lr_warmup(self):
        config = load_config(Path(__file__).parents[1] / "config" / "train.yaml")
        self.assertFalse(hasattr(config, "centering"))

        self.assertEqual(config.additional_epochs, 200)
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.precision, "bf16")
        self.assertFalse(config.use_fp16)
        self.assertEqual(config.warmup_epochs, 0)
        self.assertEqual(config.batch_size_per_gpu * config.gpu_count, 256)
        self.assertEqual(config.saveckp_freq, 50)
        self.assertEqual(config.teacher_target_cls, "centering")
        self.assertEqual(config.teacher_target_ibot, "centering")
        self.assertEqual(config.teacher_target_overlap, "sinkhorn_knopp")

    def test_per_objective_teacher_modes_are_loaded(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        values.update(
            teacher_target_cls="sinkhorn_knopp",
            teacher_target_ibot="centering",
            teacher_target_overlap="sinkhorn_knopp",
        )
        with mock.patch.object(
            Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))
        ):
            config = load_config(path)
        self.assertEqual(config.teacher_target_cls, "sinkhorn_knopp")
        self.assertEqual(config.teacher_target_ibot, "centering")
        self.assertEqual(config.teacher_target_overlap, "sinkhorn_knopp")

    def test_per_objective_teacher_modes_validate_values(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        values["teacher_target_ibot"] = "sinkhorn"
        with mock.patch.object(
            Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))
        ):
            with self.assertRaisesRegex(ValueError, "teacher_target_ibot"):
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

    def test_head_topology_must_be_boolean(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        values["shared_head"] = "false"
        with mock.patch.object(
            Path, "open", mock.mock_open(read_data=yaml.safe_dump(values))
        ):
            with self.assertRaisesRegex(ValueError, "shared_head must be a boolean"):
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

            targets, overlap_targets, overlap_patch_targets = get_teacher_targets(
                teacher,
                loss,
                epoch,
                target_modes={
                    "cls": "centering",
                    "ibot": "centering",
                    "overlap": "centering",
                },
            )

            self.assertIsNone(overlap_targets)
            self.assertIsNone(overlap_patch_targets)
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
            centered, _, _ = get_teacher_targets(
                teacher,
                centered_loss,
                epoch,
                crop_boxes=boxes,
                target_modes={
                    "cls": "centering",
                    "ibot": "centering",
                    "overlap": "centering",
                },
            )
            with mock.patch.object(
                sk_loss.region_loss, "sinkhorn_knopp_teacher",
                wraps=sk_loss.region_loss.sinkhorn_knopp_teacher,
            ) as build_overlap:
                sk, overlap, _ = get_teacher_targets(
                    teacher,
                    sk_loss,
                    epoch,
                    crop_boxes=boxes,
                    target_modes={
                        "cls": "centering",
                        "ibot": "centering",
                        "overlap": "sinkhorn_knopp",
                    },
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

        targets_a, overlap_a, _ = get_teacher_targets(
            teacher,
            first,
            0,
            crop_boxes=boxes,
            target_modes={
                "cls": "centering",
                "ibot": "centering",
                "overlap": "sinkhorn_knopp",
            },
        )
        targets_b, overlap_b, _ = get_teacher_targets(
            teacher,
            second,
            0,
            crop_boxes=boxes,
            target_modes={
                "cls": "centering",
                "ibot": "centering",
                "overlap": "sinkhorn_knopp",
            },
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
        modes = {
            "cls": "centering",
            "ibot": "centering",
            "overlap": "sinkhorn_knopp",
        }
        centered, _, _ = get_teacher_targets(
            teacher, centered_loss, 0, target_modes=modes
        )
        sk, overlap, _ = get_teacher_targets(
            teacher, sk_loss, 0, target_modes=modes
        )

        expected = centered_loss(student, centered, None, masks, None)
        actual = sk_loss(student, sk, None, masks, None, teacher_overlap_targets=overlap)

        self.assertIsNone(overlap)
        sk_loss.region_loss.sinkhorn_knopp_teacher.assert_not_called()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_all_sinkhorn_targets_skip_centering_and_centers(self):
        loss = self._loss()
        teacher = (torch.randn(4, 3) * 0.1, torch.randn(4, 4, 5) * 0.1)
        masks = [
            torch.tensor(
                [[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.bool
            ),
            torch.tensor(
                [[[1, 1], [0, 0]], [[0, 1], [1, 0]]], dtype=torch.bool
            ),
        ]
        boxes = torch.tensor(
            [[[0., 0., 1., 1., 0.], [0.25, 0., 1., 1., 0.]]] * 2
        )
        old_cls_center = loss.center.clone()
        old_patch_center = loss.center2.clone()
        with mock.patch.object(
            loss, "softmax_center_teacher_cls", wraps=loss.softmax_center_teacher_cls
        ) as center_cls, mock.patch.object(
            loss, "softmax_center_teacher_patch", wraps=loss.softmax_center_teacher_patch
        ) as center_patch, mock.patch.object(
            loss, "update_center", wraps=loss.update_center
        ) as update_center:
            targets, overlap, overlap_patch = get_teacher_targets(
                teacher,
                loss,
                0,
                crop_boxes=boxes,
                masks=masks,
                target_modes={
                    "cls": "sinkhorn_knopp",
                    "ibot": "sinkhorn_knopp",
                    "overlap": "sinkhorn_knopp",
                },
            )

        center_cls.assert_not_called()
        center_patch.assert_not_called()
        update_center.assert_not_called()
        torch.testing.assert_close(loss.center, old_cls_center)
        torch.testing.assert_close(loss.center2, old_patch_center)
        self.assertIsNotNone(overlap)
        self.assertIsNone(overlap_patch)
        torch.testing.assert_close(
            targets[0].sum(-1), torch.ones(targets[0].shape[:-1])
        )
        flat_targets = targets[1].flatten(0, 1)
        flat_masks = torch.cat([mask.flatten(1) for mask in masks], dim=0).flatten()
        torch.testing.assert_close(
            flat_targets[flat_masks].sum(-1),
            torch.ones(flat_masks.sum(), dtype=flat_targets.dtype),
        )
        self.assertEqual(flat_targets[~flat_masks].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
