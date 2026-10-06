"""Optimizer windows and teacher centers for accumulated iBOT training."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class AccumulationWindow:
    step: int
    size: int
    first: bool
    last: bool


def accumulation_window(iteration, batches, steps):
    """Include the final short window, scaling its gradients by its real size."""
    start = iteration // steps * steps
    size = min(steps, batches - start)
    return AccumulationWindow(iteration // steps, size,
                              iteration == start, iteration == start + size - 1)


class AccumulatedTeacherCenters:
    """Hold centers fixed throughout a window; update from all its teacher logits.

    Only detached FP32 sums are retained, never the patch logits or graphs.
    One distributed reduction and one center EMA are performed per window.
    Epoch checkpoints therefore need no pending accumulation state.
    """

    def __init__(self, loss):
        self.loss = loss
        self.sums = None
        self.count = 0

    @torch.no_grad()
    def add(self, cls, patch):
        values = (cls.float().sum(0, keepdim=True),
                  patch.float().mean(1).sum(0, keepdim=True))
        if self.sums is None:
            self.sums = list(values)
        else:
            for total, value in zip(self.sums, values):
                total.add_(value)
        self.count += len(cls)

    @torch.no_grad()
    def flush(self):
        if self.sums is None:
            raise RuntimeError("No teacher logits accumulated for this optimizer window")
        count = self.sums[0].new_tensor(float(self.count))
        if dist.is_available() and dist.is_initialized():
            for value in self.sums:
                dist.all_reduce(value)
            dist.all_reduce(count)
        for center, total, momentum in zip(
            (self.loss.center, self.loss.center2), self.sums,
            (self.loss.center_momentum, self.loss.center_momentum2),
        ):
            center.mul_(momentum).add_(total.reshape_as(center) / count,
                                       alpha=1 - momentum)
        self.sums = None
        self.count = 0
