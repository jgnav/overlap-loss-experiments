import copy
import unittest

import torch
from torch import nn

from model.head import iBOTHead
from tests.test_checkpoint import make_loss
from utils.checkpoint import load_continuation_state, load_resume_state
from utils.training import get_params_groups


def model(shared, bottleneck=2, layers=2, wrapped=False):
    result = nn.Module()
    result.head = iBOTHead(
        4, 3, patch_out_dim=3, hidden_dim=6, bottleneck_dim=bottleneck,
        nlayers=layers, shared_head=shared, norm_last_layer=False,
    )
    if wrapped:
        wrapper = nn.Module()
        wrapper.module = result
        return wrapper
    return result


class HeadCheckpointTest(unittest.TestCase):
    def test_shared_checkpoint_duplicates_full_heads_and_adam_state(self):
        for bottleneck, layers in ((2, 2), (2, 1), (0, 2), (0, 1)):
            with self.subTest(bottleneck=bottleneck, layers=layers):
                source = model(True, bottleneck, layers, wrapped=True)
                teacher = model(True, bottleneck, layers)
                optimizer = torch.optim.AdamW(get_params_groups(source), lr=0.002)
                tokens = torch.randn(2, 5, 4)
                outputs = source.module.head(tokens)
                sum(x.square().sum() for x in outputs).backward()
                optimizer.step()
                checkpoint = {
                    "student": source.state_dict(), "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(), "ibot_loss": make_loss().state_dict(),
                    "epoch": 8,
                }
                target = model(False, bottleneck, layers, wrapped=True)
                target_teacher = model(False, bottleneck, layers)
                target_optimizer = torch.optim.AdamW(get_params_groups(target))
                self.assertTrue(load_continuation_state(
                    checkpoint, target, target_teacher, make_loss(), target_optimizer,
                ))
                for original, converted in ((source.module.head, target.module.head),
                                            (teacher.head, target_teacher.head)):
                    for expected, actual in zip(original(tokens), converted(tokens)):
                        torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(converted(tokens[:, 0]), original(tokens[:, 0]))
                source_params = dict(source.named_parameters())
                target_params = dict(target.named_parameters())
                for name, parameter in target_params.items():
                    source_name = name.replace('patch_mlp', 'mlp').replace('last_layer2', 'last_layer')
                    if '.mlp2.' in source_name:
                        suffix = f'mlp.{len(source.module.head.mlp) - 1}' if layers > 1 else 'mlp'
                        source_name = source_name.replace('mlp2', suffix)
                    original = source_params[source_name]
                    for key in ('step', 'exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(target_optimizer.state[parameter][key], optimizer.state[original][key])
                head = target.module.head
                target_optimizer.zero_grad(set_to_none=True)
                before = {name: p.detach().clone() for name, p in head.named_parameters()}
                head(tokens)[1].square().sum().backward()
                self.assertTrue(all(p.grad is None for p in head.mlp.parameters()))
                self.assertTrue(any(p.grad is not None for p in head.mlp2.parameters()) if not bottleneck
                                else any(p.grad is not None for p in head.patch_mlp.parameters()))
                target_optimizer.step()
                for name, p in head.mlp.named_parameters():
                    torch.testing.assert_close(p, before['mlp.' + name])

                # A subsequent exact resume preserves independently trained branches.
                resumed = model(False, bottleneck, layers, wrapped=True)
                resumed_optimizer = torch.optim.AdamW(get_params_groups(resumed))
                saved = {**checkpoint, 'student': copy.deepcopy(target.state_dict()),
                         'teacher': target_teacher.state_dict(), 'optimizer': target_optimizer.state_dict()}
                self.assertEqual(load_resume_state(saved, resumed, model(False, bottleneck, layers),
                                                   make_loss(), resumed_optimizer, None), 8)
                for a, b in zip(target.module.head(tokens), resumed.module.head(tokens)):
                    torch.testing.assert_close(a, b)

    def test_shared_head_reuses_parameters_for_both_token_types(self):
        head = model(True).head
        self.assertIs(head.last_layer, head.last_layer2)
        tokens = torch.randn(2, 5, 4)
        head(tokens)[1].sum().backward()
        self.assertTrue(all(p.grad is not None for p in head.mlp.parameters()))


if __name__ == '__main__':
    unittest.main()
