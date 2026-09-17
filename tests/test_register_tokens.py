import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import yaml

from evaluation.utils import common
from losses import iBOTLoss
from model.vision_transformer import VisionTransformer
from train import load_config
from utils.checkpoint import load_continuation_state, load_resume_state
from utils.training import MultiCropWrapper


def make_loss():
    return iBOTLoss(
        out_dim=3, patch_out_dim=3, ngcrops=2, nlcrops=0,
        warmup_teacher_temp=.07, teacher_temp=.07,
        warmup_teacher_temp2=.07, teacher_temp2=.07,
        warmup_teacher_temp_epochs=0, nepochs=1,
    )


class RecordingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, tokens):
        self.seen = tokens
        return tokens[:, 0], tokens[:, 1:]


class TinyBackbone(nn.Module):
    def __init__(self, registers):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2))
        self.register_tokens = (
            nn.Parameter(torch.randn(1, registers, 2)) if registers else None
        )


class TinyWrapper(nn.Module):
    def __init__(self, registers):
        super().__init__()
        self.backbone = TinyBackbone(registers)


class RegisterTokenTest(unittest.TestCase):
    def test_zero_registers_preserves_original_sequence_and_state_dict(self):
        model = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=12, depth=1,
            num_heads=3, num_register_tokens=0,
        ).eval()
        with torch.no_grad():
            tokens = model(torch.randn(2, 3, 32, 32), return_all_tokens=True)
        self.assertEqual(tokens.shape, (2, 5, 12))
        self.assertIsNone(model.register_tokens)
        self.assertNotIn("register_tokens", model.state_dict())

    def test_registers_are_between_cls_and_patches_without_positions(self):
        model = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=4, depth=0,
            num_heads=1, num_register_tokens=4,
        ).eval()
        with torch.no_grad():
            model.patch_embed.proj.weight.zero_()
            model.patch_embed.proj.bias.zero_()
            model.cls_token.fill_(2)
            model.pos_embed.zero_()
            model.pos_embed[:, 0].fill_(3)
            model.pos_embed[:, 1:].fill_(5)
            model.register_tokens.fill_(7)
            tokens = model.prepare_tokens(torch.zeros(1, 3, 32, 32))
        self.assertEqual(tokens.shape, (1, 9, 4))
        torch.testing.assert_close(tokens[:, 0], torch.full((1, 4), 5.0))
        torch.testing.assert_close(tokens[:, 1:5], torch.full((1, 4, 4), 7.0))
        torch.testing.assert_close(tokens[:, 5:], torch.full((1, 4, 4), 5.0))

    def test_heads_and_intermediate_features_exclude_registers(self):
        backbone = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=12, depth=1,
            num_heads=3, return_all_tokens=True, num_register_tokens=4,
        )
        head = RecordingHead()
        wrapped = MultiCropWrapper(backbone, head)
        image = torch.randn(2, 3, 32, 32)
        cls, patches = wrapped(image)
        self.assertEqual(head.seen.shape, (2, 5, 12))
        self.assertEqual(cls.shape, (2, 12))
        self.assertEqual(patches.shape, (2, 4, 12))
        self.assertEqual(backbone.get_intermediate_layers(image, n=1)[0].shape,
                         (2, 5, 12))
        (cls.sum() + patches.sum()).backward()
        self.assertIsNotNone(backbone.register_tokens.grad)

    def test_config_accepts_nonnegative_integer_register_count(self):
        path = Path(__file__).parents[1] / "config" / "train.yaml"
        values = yaml.safe_load(path.read_text())
        for count in (0, 4):
            with mock.patch.object(
                Path, "open", mock.mock_open(read_data=yaml.safe_dump({**values, "register": count}))
            ):
                self.assertEqual(load_config(path).register, count)
        for count in (-1, 1.5, True):
            with mock.patch.object(
                Path, "open", mock.mock_open(read_data=yaml.safe_dump({**values, "register": count}))
            ), self.assertRaisesRegex(ValueError, "register"):
                load_config(path)

    def test_continuation_adds_registers_but_exact_resume_requires_them(self):
        source_student, source_teacher = TinyWrapper(0), TinyWrapper(0)
        source_optimizer = torch.optim.AdamW(source_student.parameters())
        source_student.backbone.weight.sum().backward()
        source_optimizer.step()
        checkpoint = {
            "student": source_student.state_dict(),
            "teacher": source_teacher.state_dict(),
            "ibot_loss": make_loss().state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "epoch": 0,
        }
        student, teacher = TinyWrapper(4), TinyWrapper(4)
        student_registers = student.backbone.register_tokens.detach().clone()
        optimizer = torch.optim.AdamW(student.parameters())
        restored = load_continuation_state(
            checkpoint, student, teacher, make_loss(), optimizer
        )
        self.assertTrue(restored)
        torch.testing.assert_close(student.backbone.register_tokens, student_registers)
        torch.testing.assert_close(
            teacher.backbone.register_tokens, student.backbone.register_tokens
        )
        self.assertNotIn(student.backbone.register_tokens, optimizer.state)
        with self.assertRaisesRegex(ValueError, "register_tokens"):
            load_resume_state(
                checkpoint, TinyWrapper(4), TinyWrapper(4), make_loss(),
                torch.optim.AdamW(TinyWrapper(4).parameters()), None,
            )

    def test_evaluation_reconstructs_register_count_from_checkpoint(self):
        def factory(_architecture, **kwargs):
            return VisionTransformer(embed_dim=12, depth=1, num_heads=3, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            source = factory(
                "vit_small", img_size=[64], patch_size=16,
                num_register_tokens=4, return_all_tokens=True,
            )
            path = Path(directory) / "checkpoint.pth"
            torch.save({"teacher": source.state_dict(), "args": {"arch": "vit_small"}}, path)
            with mock.patch.object(common, "create_model", side_effect=factory):
                loaded, metadata = common.load_backbone(path)
            self.assertEqual(loaded.num_register_tokens, 4)
            self.assertEqual(metadata["num_register_tokens"], 4)
            torch.testing.assert_close(loaded.register_tokens, source.register_tokens)


if __name__ == "__main__":
    unittest.main()
