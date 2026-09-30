"""Stable, detached Sinkhorn assignments for selected teacher patches."""

import torch
import torch.distributed as dist


@torch.no_grad()
def sinkhorn_log_probabilities(logits, temperature, iterations=3):
    """Joint assignments for [selected patches, prototypes] across all ranks.

    Alternate global prototype balancing and per-patch normalization. Return
    log probabilities with each patch summing to one. Callers select the
    overlapping teacher patches before normalization; students use softmax.
    Every rank must participate, including ranks with no selected patches.
    """
    distributed = dist.is_available() and dist.is_initialized()
    count = logits.new_tensor(logits.shape[0], dtype=torch.float32)
    if distributed:
        dist.all_reduce(count)
    log_q = logits.float() / temperature
    if count.item() == 0:
        return log_q
    for _ in range(iterations):
        # A detached shift stabilizes exp without changing its derivatives.
        maximum = (log_q.detach().amax(dim=0) if len(log_q)
                   else log_q.new_full((log_q.shape[1],), -torch.inf))
        if distributed:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        total = (log_q - maximum).exp().sum(dim=0)
        if distributed:
            dist.all_reduce(total)
        log_q = log_q - (maximum + total.log())
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
    return log_q
