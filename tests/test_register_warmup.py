import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn
import yaml

from model.vision_transformer import VisionTransformer
from train import (
    configure_slurm_requeue_resume,
    ensure_training_wandb,
    init_wandb,
    load_config,
    log_training_epoch_to_wandb,
)
from utils.checkpoint import load_resume_state, read_resume_checkpoint
from utils.register_warmup import (
    clear_non_register_gradients,
    completed_normal_training_epochs,
    prepend_register_warmup,
    teacher_ema_pairs,
)
from utils.training import MultiCropWrapper, cosine_scheduler


ROOT = Path(__file__).parents[1]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.weight = nn.Parameter(torch.ones(2, 2))
        self.backbone.register_tokens = nn.Parameter(torch.ones(1, 4, 2))
        self.head = nn.Linear(2, 2)

    def loss(self):
        return sum(parameter.square().sum() for parameter in self.parameters())


class TokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 3)

    def forward(self, tokens):
        return self.linear(tokens[:, 0]), self.linear(tokens[:, 1:])


class TinyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, 2))
        self.register_buffer("center2", torch.zeros(1, 2))


class RegisterWarmupTest(unittest.TestCase):
    def wandb_args(self):
        return SimpleNamespace(
            register_warmup_epochs=5, epochs=55, normal_training_epochs=50,
            effective_batch_size=256, source_checkpoint_epoch=800,
            source_equivalent_final_epoch=855,
        )

    def test_wandb_initialization_waits_for_warmup_and_resumes_existing_run(self):
        args = self.wandb_args()
        run, initialize = mock.Mock(), mock.Mock()
        initialize.return_value = run
        for epoch in range(5):
            self.assertIsNone(ensure_training_wandb(args, epoch, 0, None, initialize))
        initialize.assert_not_called()
        self.assertIs(ensure_training_wandb(args, 5, 0, None, initialize), run)
        initialize.assert_called_once_with()
        self.assertEqual(run.config.update.call_args.args[0]["additional_epochs"], 50)
        self.assertEqual(run.config.update.call_args.args[0]["total_stage_epochs"], 55)
        ensure_training_wandb(args, 7, 7, run, initialize)
        initialize.assert_called_once_with()
        self.assertEqual(run.config.update.call_args.args[0]["continuation_start_epoch"], 2)

    def test_wandb_training_rows_exclude_warmup_and_use_zero_based_continuation(self):
        args, run = self.wandb_args(), mock.Mock()
        for epoch in range(55):
            log_training_epoch_to_wandb(run, {"loss": .25}, args, epoch, 10)
        self.assertEqual(run.log.call_count, 50)
        rows = [call.args[0] for call in run.log.call_args_list]
        self.assertEqual([row["epoch"] for row in rows], list(range(50)))
        self.assertEqual(rows[0]["state/global_step"], 10)
        self.assertEqual(rows[-1]["state/global_step"], 500)
        self.assertEqual(rows[0]["state/total_stage_epoch"], 6)
        self.assertEqual(rows[-1]["state/continuation_epoch"], 50)
        self.assertTrue(all(row["state/register_warmup_active"] == 0 for row in rows))
        self.assertTrue(all(row["train/loss"] == .25 for row in rows))
        # An ordinary ablation retains the existing W&B epoch convention.
        args.register_warmup_epochs = 0
        run.reset_mock()
        log_training_epoch_to_wandb(run, {"loss": .25}, args, 0, 10)
        self.assertEqual(run.log.call_args.args[0]["epoch"], 1)

    def test_wandb_config_describes_fifty_normal_epochs(self):
        args = self.wandb_args()
        with mock.patch("train.init_wandb_run") as initialize:
            init_wandb(args)
        config = initialize.call_args.args[1]
        self.assertEqual((config["epochs"], config["additional_epochs"],
                          config["total_stage_epochs"]), (50, 50, 55))
        self.assertEqual(args.epochs, 55)

    def test_slurm_restart_before_wandb_exists_resumes_warmup_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "checkpoint.pth").touch()
            args = SimpleNamespace(
                output_dir=str(path), run_id="new-register", resume_checkpoint=None,
                reset_optimizer=True, wandb_mode="online", wandb_run_id=None,
                wandb_resume=None, register_warmup_epochs=5,
            )
            with mock.patch.dict("os.environ", {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "1"}):
                configure_slurm_requeue_resume(args)
            self.assertEqual(args.resume_checkpoint, (path / "checkpoint.pth").resolve())
            self.assertFalse(args.reset_optimizer)
            self.assertIsNone(args.wandb_run_id)
            self.assertIsNone(args.wandb_resume)

    def test_warmup_preserves_pretrained_weights_and_existing_adamw_moments(self):
        student = TinyModel()
        optimizer = torch.optim.AdamW(student.parameters(), lr=.01, weight_decay=.5)
        # Populate AdamW moments, as happens when continuing an existing source.
        student.loss().backward()
        optimizer.step()
        frozen = {
            name: (parameter.detach().clone(), copy.deepcopy(optimizer.state[parameter]))
            for name, parameter in student.named_parameters()
            if name != "backbone.register_tokens"
        }
        registers_before = student.backbone.register_tokens.detach().clone()
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            student.loss().backward()
            clear_non_register_gradients(student)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
        self.assertFalse(torch.equal(registers_before, student.backbone.register_tokens))
        for name, parameter in student.named_parameters():
            if name in frozen:
                value, state = frozen[name]
                self.assertTrue(torch.equal(parameter, value), name)
                for key, expected in state.items():
                    self.assertTrue(torch.equal(optimizer.state[parameter][key], expected),
                                    f"{name}: {key}")
        # Normal training immediately enables the backbone and projection head.
        optimizer.zero_grad(set_to_none=True)
        student.loss().backward()
        optimizer.step()
        for name, parameter in student.named_parameters():
            if name in frozen:
                self.assertFalse(torch.equal(parameter, frozen[name][0]), name)

    def test_registers_receive_gradients_through_the_frozen_transformer_and_head(self):
        torch.manual_seed(17)
        backbone = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=12, depth=1,
            num_heads=3, return_all_tokens=True, num_register_tokens=4,
        )
        student = MultiCropWrapper(backbone, TokenHead()).eval()
        optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
        before = {name: parameter.detach().clone()
                  for name, parameter in student.named_parameters()}
        cls, patches = student(torch.randn(2, 3, 32, 32))
        (cls.square().sum() + patches.square().sum()).backward()
        clear_non_register_gradients(student)
        self.assertGreater(backbone.register_tokens.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, parameter in student.named_parameters():
            if name == "backbone.register_tokens":
                self.assertFalse(torch.equal(parameter, before[name]))
            else:
                self.assertIsNone(parameter.grad, name)
                self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_teacher_ema_only_updates_registers_during_warmup(self):
        student, teacher = TinyModel(), TinyModel()
        with torch.no_grad():
            for parameter in teacher.parameters():
                parameter.fill_(7)
        before = {name: parameter.detach().clone()
                  for name, parameter in teacher.named_parameters()}
        with torch.no_grad():
            for source, target in teacher_ema_pairs(student, teacher, register_only=True):
                target.mul_(.9).add_(source, alpha=.1)
        for name, parameter in teacher.named_parameters():
            if name == "backbone.register_tokens":
                torch.testing.assert_close(parameter, before[name] * .9 +
                                           student.backbone.register_tokens * .1)
            else:
                self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assertEqual(len(teacher_ema_pairs(student, teacher)),
                         len(list(teacher.parameters())))

    def test_fifty_epoch_schedules_and_probe_labels_start_after_warmup(self):
        normal = cosine_scheduler(.1, .001, 50, 2)
        actual = prepend_register_warmup(normal, 5, 2)
        self.assertEqual(len(actual), 55 * 2)
        np.testing.assert_array_equal(actual[:10], np.full(10, normal[0]))
        np.testing.assert_array_equal(actual[10:], normal)
        self.assertIs(prepend_register_warmup(normal, 0, 2), normal)
        probe_epochs = [(total, completed_normal_training_epochs(total, 5))
                        for total in range(1, 56)
                        if completed_normal_training_epochs(total, 5) > 0
                        and completed_normal_training_epochs(total, 5) % 5 == 0]
        self.assertEqual(probe_epochs, [(epoch + 5, epoch) for epoch in range(5, 51, 5)])

    def test_only_register_ablation_enables_warmup(self):
        for path in sorted((ROOT / "config" / "ablations").glob("*.yaml")):
            with self.subTest(config=path.name):
                args = load_config(path)
                if path.name == "register_4.yaml":
                    self.assertEqual((args.register_warmup_epochs, args.epochs,
                                      args.normal_training_epochs), (5, 55, 50))
                else:
                    self.assertEqual(args.register_warmup_epochs, 0)
                    self.assertEqual(args.normal_training_epochs, args.epochs)

    def test_invalid_warmup_configurations_are_rejected(self):
        path = ROOT / "config" / "ablations" / "register_4.yaml"
        values = yaml.safe_load(path.read_text())
        invalid = [{"register_warmup_epochs": value} for value in (-1, 1.5, True, 55)]
        invalid.append({"register": 0})
        for overrides in invalid:
            with self.subTest(overrides=overrides), mock.patch.object(
                Path, "open", mock.mock_open(read_data=yaml.safe_dump({**values, **overrides}))
            ), self.assertRaisesRegex(ValueError, "register_warmup_epochs"):
                load_config(path)

    def test_full_resume_restores_warmup_and_normal_training_phases(self):
        for completed in (3, 7):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as directory:
                student, teacher, loss = TinyModel(), TinyModel(), TinyLoss()
                optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
                for epoch in range(completed):
                    optimizer.zero_grad(set_to_none=True)
                    student.loss().backward()
                    if epoch < 5:
                        clear_non_register_gradients(student)
                    optimizer.step()
                path = Path(directory) / "checkpoint.pth"
                torch.save({
                    "student": student.state_dict(), "teacher": teacher.state_dict(),
                    "ibot_loss": loss.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": completed,
                    "args": SimpleNamespace(epochs=55, register_warmup_epochs=5,
                                            use_fp16=False, source_checkpoint_epoch=800),
                }, path)
                args = SimpleNamespace(resume_checkpoint=path, epochs=55,
                                       register_warmup_epochs=5, use_fp16=False)
                checkpoint = read_resume_checkpoint(args)
                restored, restored_teacher = TinyModel(), TinyModel()
                restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.)
                start = load_resume_state(checkpoint, restored, restored_teacher,
                                          TinyLoss(), restored_optimizer, None)
                self.assertEqual(start, completed)
                self.assertEqual(args.resume_epoch, completed)
                self.assertEqual(args.source_checkpoint_epoch, 800)
                # Compare the next update against uninterrupted execution.
                for model, opt in ((student, optimizer), (restored, restored_optimizer)):
                    opt.zero_grad(set_to_none=True)
                    model.loss().backward()
                    if start < 5:
                        clear_non_register_gradients(model)
                    opt.step()
                for expected, actual in zip(student.parameters(), restored.parameters()):
                    self.assertTrue(torch.equal(actual, expected))
                # A previous run's zero warmup cannot be relabeled as this experiment.
                del checkpoint["args"].register_warmup_epochs
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "register_warmup_epochs"):
                    read_resume_checkpoint(args)
                args.register_warmup_epochs = 0
                read_resume_checkpoint(args)


if __name__ == "__main__":
    unittest.main()
