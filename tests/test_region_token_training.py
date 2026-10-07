"""Exercise the actual trainer's regional-token return paths on CPU."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from tests.test_gradient_accumulation import TinyDDP
from tests.test_ibot_loss import make_loss
from tests.test_region_loss import boxes_full, boxes_disjoint
from tests.test_region_token import model
from train import train_one_epoch
from utils.training import get_params_groups


class RegionTokenTrainingTest(unittest.TestCase):
    def test_trainer_updates_token_with_and_without_backbone_feature_returns(self):
        for koleo in (False, True):
            with self.subTest(koleo=koleo):
                torch.manual_seed(17)
                student = TinyDDP(model(masked=True))
                teacher = model().requires_grad_(False)
                teacher.load_state_dict(student.module.state_dict(), strict=False)
                loss = make_loss(region_aggregation='region_token', region_normalization='centering',
                                 nlcrops=1, koleo_regularizer=koleo)
                optimizer = torch.optim.AdamW(get_params_groups(student), lr=.002)
                before = student.module.region_token.token.detach().clone()
                batches = []
                for _ in range(2):
                    images = [torch.randn(2, 3, size, size) for size in (32, 32, 16)]
                    masks = [torch.tensor([[[True, False], [False, True]]]).expand(2, -1, -1)
                             for _ in range(2)]
                    boxes = boxes_full(2)
                    boxes[1:] = boxes_disjoint()
                    batches.append((images, None, masks, boxes))
                args = SimpleNamespace(
                    diagnostic_max_patch_features_per_batch=64, source_checkpoint_epoch=800,
                    epochs=2, register_warmup_epochs=0, gradient_accumulation_steps=1,
                    print_freq=100, precision='fp32', diagnostic_feature_batches=1,
                    global_crops_number=2, local_crops_number=1, clip_grad=0, freeze_last_layer=0,
                )
                with mock.patch.object(torch.Tensor, 'cuda', lambda x, **kw: x), \
                     mock.patch('torch.cuda.synchronize'), \
                     mock.patch('train.utils.concat_all_gather', lambda x: x):
                    stats = train_one_epoch(
                        student, teacher, teacher, loss, batches, optimizer,
                        np.array([.002, .001]), np.array([.04, .04]), np.array([.9, .9]),
                        0, None, args,
                    )
                self.assertTrue(np.isfinite(stats['loss']))
                self.assertEqual(stats['region_valid_ratio'], .5)
                self.assertFalse(torch.equal(before, student.module.region_token.token))
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))


if __name__ == '__main__':
    unittest.main()
