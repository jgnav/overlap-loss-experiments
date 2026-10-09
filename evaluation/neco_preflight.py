"""Smoke test the released NeCo losses, metrics and checkpoint adapter on GPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor/neco'))
import torch
from experiments.linear_segmentation.linear_finetune import LinearFinetune
from experiments.utils import PredsmIoU
from evaluation.neco_benchmarks import strict_neco_model


if __name__ == '__main__':
    torch.set_num_threads(4)
    checkpoint = sys.argv[1]
    model, state, metadata = strict_neco_model(checkpoint)
    probe = LinearFinetune(patch_size=16, num_classes=12, lr=.01,
                          input_size=448, spatial_res=28, val_iters=512,
                          drop_at=20, arch='vit-small', arch_version='v1', head_type='lc')
    msg = probe.load_state_dict({'model.' + k: v for k, v in state.items()}, strict=False)
    assert all(k.startswith(('model.projection_head.', 'finetune_head.')) for k in msg.missing_keys)
    assert not msg.unexpected_keys
    probe.cuda()
    # Test the actual released batch size and loss/backward implementation.
    images = torch.randn(128, 3, 448, 448, device='cuda')
    masks = torch.randint(0, 12, (128, 1, 448, 448), device='cuda').float() / 255
    loss = probe.training_step((images, masks), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert probe.finetune_head.weight.grad is not None
    assert all(p.grad is None for p in probe.model.parameters())
    metric = PredsmIoU(3, 3)
    metric.update(torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([2, 2, 0, 0, 1, 1]))
    value = metric.compute(True)[0]
    assert abs(value - 1) < 1e-8, value
    print('NECO_GPU_PREFLIGHT_OK', metadata, 'loss', loss.item(),
          'peak_GPU_GiB', torch.cuda.max_memory_allocated()/2**30, flush=True)
