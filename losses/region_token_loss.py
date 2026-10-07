"""Student/EMA-teacher cross-view distillation of learned overlap tokens."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .region_loss import RegionLoss
from .sinkhorn import sinkhorn_log_probabilities


class RegionTokenLoss(RegionLoss):
    def __init__(self, *args, out_dim, center_momentum=.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_momentum = center_momentum
        if self.normalization == "centering":
            # Concept logits have their own statistics, separate from iBOT.
            self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits, teacher_logits, crop_boxes, *, patch_count):
        if (len(student_logits) != 2 or len(teacher_logits) != 2
                or student_logits[0].ndim != 2
                or student_logits[0].shape[0] != len(crop_boxes)
                or any(x.shape != student_logits[0].shape
                       for x in (*student_logits, *teacher_logits))):
            raise ValueError("Region token logits must match [batch, prototypes] for two views")
        geometry = self.prepare_geometry(crop_boxes, patch_count)
        valid = geometry["valid"]
        count = valid.sum().float()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            world_size = dist.get_world_size()
        teacher = torch.stack(teacher_logits, dim=1).detach().float()[valid]
        if self.normalization == "sinkhorn" and count.item() > 0:
            targets = sinkhorn_log_probabilities(
                teacher.flatten(0, 1), self.temperature
            ).exp().reshape_as(teacher)
        else:
            centered = teacher - self.center if self.normalization == "centering" else teacher
            targets = (centered / self.temperature).softmax(-1)
        if valid.any():
            student = torch.stack(student_logits, dim=1).float()[valid]
            log_probs = F.log_softmax(student / self.student_temperature, dim=-1)
            local_sum = -.5 * (
                (targets[:, 0] * log_probs[:, 1]).sum()
                + (targets[:, 1] * log_probs[:, 0]).sum()
            )
        else:
            local_sum = sum(x.float().sum() * 0 for x in student_logits)
        if self.normalization == "centering":
            with torch.no_grad():
                total = teacher.sum((0, 1), keepdim=False)[None]
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(total)
                if count.item() > 0:
                    self.center.mul_(self.center_momentum).add_(
                        total / (2 * count), alpha=1 - self.center_momentum
                    )
        return {
            "loss": local_sum * world_size / count.clamp_min(1),
            "valid_ratio": valid.float().mean(),
            "intersection_area": geometry["intersection_area"].mean(),
            "patch_mask": geometry["selected"],
            "patch_weights": geometry["weights"],
            "valid": valid,
        }
