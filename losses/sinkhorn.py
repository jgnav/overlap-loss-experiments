import torch
import torch.distributed as dist


@torch.no_grad()
def sinkhorn_knopp(teacher_logits, teacher_temp, n_iterations=3):
    """DINOv2 assignment normalization on raw [tokens, prototypes] logits.

    Count tokens across ranks as in DINOv2's patch loss: overlap selection
    can produce different counts, including zero, on individual ranks.
    No centering or softmax is applied before this transformation.
    """
    assignments = (teacher_logits.float() / teacher_temp).exp().t()
    distributed = dist.is_available() and dist.is_initialized()
    token_count = torch.tensor(
        teacher_logits.shape[0], device=teacher_logits.device, dtype=torch.long
    )
    if distributed:
        dist.all_reduce(token_count)
    if token_count.item() == 0:
        return assignments.t()

    total_mass = assignments.sum()
    if distributed:
        dist.all_reduce(total_mass)
    assignments /= total_mass
    prototype_count = assignments.shape[0]
    for _ in range(n_iterations):
        prototype_mass = assignments.sum(dim=1, keepdim=True)
        if distributed:
            dist.all_reduce(prototype_mass)
        assignments /= prototype_mass
        assignments /= prototype_count
        assignments /= assignments.sum(dim=0, keepdim=True)
        assignments /= token_count
    assignments *= token_count
    return assignments.t()
