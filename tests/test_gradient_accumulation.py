"""Check accumulated updates against full batches through the actual trainer."""

import copy
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from losses import iBOTLoss
from train import get_teacher_targets, load_config, train_one_epoch
from utils import training as utils
from utils.checkpoint import _validate_resume_compatibility
from utils.gradient_accumulation import AccumulatedTeacherCenters, accumulation_window


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.backbone.embed_dim = 3
        self.backbone.num_register_tokens = 0
        self.backbone.masked_im_modeling = False

    def forward(self, images, return_backbone_feat=False, **kwargs):
        tokens = self.backbone(torch.cat(images))
        output = (tokens[:, 0], tokens[:, 1:])
        return (tokens, output) if return_backbone_feat else output


class TinyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.suppressed = 0
        self.in_no_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        self.suppressed += 1
        self.in_no_sync = True
        try:
            yield
        finally:
            self.in_no_sync = False


def make_loss():
    return iBOTLoss(3, 3, 2, 0, .07, .07, .07, .07, 0, 2,
                    lambda3=.4, region_normalization="centering")


def distributed_accumulation_worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(9)
        model = nn.parallel.DistributedDataParallel(nn.Linear(3, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        loss = make_loss()
        centers = AccumulatedTeacherCenters(loss)
        torch.manual_seed(100 + rank)
        data = torch.randn(5, 2, 3)
        for i, x in enumerate(data):
            window = accumulation_window(i, len(data), 3)
            if window.first:
                optimizer.zero_grad()
            context = model.no_sync() if not window.last else nullcontext()
            with context:
                (model(x).square().mean() / window.size).backward()
            centers.add(x, x[:, None].expand(-1, 4, -1))
            if window.last:
                optimizer.step()
                centers.flush()
        if rank == 0:
            torch.save({"model": model.module.state_dict(), "centers": loss.state_dict()}, destination)
    finally:
        dist.destroy_process_group()


class AccumulationTest(unittest.TestCase):
    def test_two_rank_ddp_and_distributed_centers_match_global_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            result = str(Path(directory) / "result.pth")
            mp.spawn(distributed_accumulation_worker,
                     args=(str(Path(directory) / "rendezvous"), result), nprocs=2, join=True)
            actual = torch.load(result, weights_only=True)
        torch.manual_seed(9)
        model = nn.Linear(3, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        loss = make_loss()
        data = []
        for rank in range(2):
            torch.manual_seed(100 + rank)
            data.append(torch.randn(5, 2, 3))
        for start, end in [(0, 3), (3, 5)]:
            x = torch.cat([r[start:end].flatten(0, 1) for r in data])
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
            loss.update_center(x, x[:, None].expand(-1, 4, -1))
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(actual["model"][name], expected)
        for name, expected in loss.state_dict().items():
            torch.testing.assert_close(actual["centers"][name], expected)

    def test_configs_preserve_microbatch_and_actual_lr(self):
        root = Path(__file__).resolve().parents[1]
        for model, micro, steps, total, peak in [
            ("base", 48, 3, 576, 7.5e-6), ("large", 32, 4, 512, 5e-6),
        ]:
            args = load_config(root / f"config/long_training/ibot_vit_{model}.yaml")
            self.assertEqual(args.batch_size_per_gpu, micro)
            self.assertEqual(args.gradient_accumulation_steps, steps)
            self.assertEqual(args.gpu_count * micro * steps, total)
            self.assertAlmostEqual(args.lr * total / args.reference_batch_size, peak)
            self.assertEqual(args.precision, "bf16")
            self.assertIsNone(args.resume_checkpoint)

    def test_old_checkpoint_defaults_to_one_step_and_rejects_accumulation_change(self):
        _validate_resume_compatibility({"args": {}}, SimpleNamespace(gradient_accumulation_steps=1))
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            _validate_resume_compatibility({"args": {}}, SimpleNamespace(gradient_accumulation_steps=3))

    def test_teacher_centers_use_the_full_window_and_stay_fixed_for_targets(self):
        loss = make_loss()
        acc = AccumulatedTeacherCenters(loss)
        cls, patch = torch.randn(6, 3), torch.randn(6, 4, 3)
        old = (loss.center.clone(), loss.center2.clone())
        for start, end in [(0, 2), (2, 6)]:
            targets = get_teacher_targets((cls[start:end], patch[start:end]), loss, 0, acc)
            torch.testing.assert_close(targets[0], ((cls[start:end] - old[0]) / .07).softmax(-1))
            torch.testing.assert_close(loss.center, old[0])
            torch.testing.assert_close(loss.center2, old[1])
        acc.flush()
        torch.testing.assert_close(loss.center, cls.mean(0, keepdim=True) * .1)
        torch.testing.assert_close(loss.center2, patch.mean((0, 1), keepdim=True) * .1)
        self.assertIsNone(acc.sums)

    def test_actual_trainer_matches_full_batches_with_a_short_final_window(self):
        torch.manual_seed(7)
        initial = TinyModel()
        batches = []
        for _ in range(5):
            images = [torch.randn(2, 5, 3), torch.randn(2, 5, 3)]
            masks = [torch.ones(2, 2, 2, dtype=torch.bool) for _ in range(2)]
            boxes = torch.tensor([[[0., 0., 1., 1., 0.]] * 2] * 2)
            batches.append((images, None, masks, boxes))
        full = []
        for start, end in [(0, 3), (3, 5)]:
            group = batches[start:end]
            full.append(([torch.cat([x[0][v] for x in group]) for v in range(2)],
                         None, [torch.cat([x[2][v] for x in group]) for v in range(2)],
                         torch.cat([x[3] for x in group])))

        def run(data, steps):
            student = TinyDDP(copy.deepcopy(initial))
            teacher = copy.deepcopy(initial)
            for p in teacher.parameters():
                p.requires_grad = False
            loss = make_loss()
            optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
            args = SimpleNamespace(
                diagnostic_max_patch_features_per_batch=64, source_checkpoint_epoch=400,
                epochs=2, register_warmup_epochs=0, gradient_accumulation_steps=steps,
                print_freq=100, precision="fp32", diagnostic_feature_batches=1,
                global_crops_number=2, local_crops_number=0, clip_grad=.3, freeze_last_layer=3,
            )
            with mock.patch.object(torch.Tensor, "cuda", lambda x, **kw: x), \
                 mock.patch("torch.cuda.synchronize"), \
                 mock.patch("train.utils.concat_all_gather", lambda x: x), \
                 mock.patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                 mock.patch("train.utils.clip_gradients", wraps=utils.clip_gradients) as clip:
                train_one_epoch(student, teacher, teacher, loss, data, optimizer,
                                np.array([.001, .0007]), np.array([.04, .08]),
                                np.array([.9, .95]), 0, None, args)
                self.assertEqual(step.call_count, 2)
                self.assertEqual(clip.call_count, 2)
            return student, teacher, loss, optimizer

        accumulated = run(batches, 3)
        reference = run(full, 1)
        self.assertEqual(accumulated[0].suppressed, 3)
        for a, b in zip(accumulated[:3], reference[:3]):
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], atol=2e-6, rtol=2e-5)
        for a, b in zip(accumulated[3].state.values(), reference[3].state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
