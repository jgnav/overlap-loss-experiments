"""CUDA initialization probe launched once per local rank by torchrun."""

import os

import torch


local_rank = int(os.environ["LOCAL_RANK"])
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"rank={local_rank} cuda={available} count={count}", flush=True)
if not available:
    raise RuntimeError("CUDA is unavailable")
torch.cuda.set_device(local_rank)
torch.empty(1, device="cuda").add_(1)
torch.cuda.synchronize()
