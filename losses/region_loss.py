import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .sinkhorn import sinkhorn_knopp


@dataclass
class OverlapTeacherTargets:
    probabilities: torch.Tensor  # [batch, 2 views, prototypes], pooled after SK
    patch_weights: torch.Tensor
    valid: torch.Tensor
    intersection_area: torch.Tensor


def intersection_patch_weights(crop_boxes, patch_count, min_area):
    """Return per-patch coverage of the two-view intersection.

    Crop boxes use normalized original-image coordinates in the form
    ``[left, top, right, bottom, horizontally_flipped]``.
    """
    if crop_boxes.ndim != 3 or crop_boxes.shape[1:] != (2, 5):
        raise ValueError(
            "crop_boxes must have shape [batch_size, 2, 5], got "
            f"{tuple(crop_boxes.shape)}"
        )

    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(
            f"The number of patch tokens must be square, got {patch_count}"
        )

    boxes = crop_boxes[..., :4]
    intersection = torch.stack(
        [
            torch.maximum(boxes[:, 0, 0], boxes[:, 1, 0]),
            torch.maximum(boxes[:, 0, 1], boxes[:, 1, 1]),
            torch.minimum(boxes[:, 0, 2], boxes[:, 1, 2]),
            torch.minimum(boxes[:, 0, 3], boxes[:, 1, 3]),
        ],
        dim=-1,
    )
    intersection_width = (intersection[:, 2] - intersection[:, 0]).clamp_min(0)
    intersection_height = (intersection[:, 3] - intersection[:, 1]).clamp_min(0)
    intersection_area = intersection_width * intersection_height
    valid = (intersection_area > 0) & (intersection_area >= min_area)

    crop_width = (boxes[:, :, 2] - boxes[:, :, 0]).clamp_min(1e-12)
    crop_height = (boxes[:, :, 3] - boxes[:, :, 1]).clamp_min(1e-12)
    local_left = (intersection[:, None, 0] - boxes[:, :, 0]) / crop_width
    local_top = (intersection[:, None, 1] - boxes[:, :, 1]) / crop_height
    local_right = (intersection[:, None, 2] - boxes[:, :, 0]) / crop_width
    local_bottom = (intersection[:, None, 3] - boxes[:, :, 1]) / crop_height

    flipped = crop_boxes[:, :, 4] >= 0.5
    mirrored_left = 1.0 - local_right
    mirrored_right = 1.0 - local_left
    local_left = torch.where(flipped, mirrored_left, local_left)
    local_right = torch.where(flipped, mirrored_right, local_right)
    local_boxes = torch.stack(
        [local_left, local_top, local_right, local_bottom], dim=-1
    ).clamp(0.0, 1.0)

    edges = torch.linspace(
        0.0,
        1.0,
        grid_size + 1,
        device=crop_boxes.device,
        dtype=crop_boxes.dtype,
    )
    horizontal_coverage = (
        torch.minimum(local_boxes[:, :, 2, None], edges[1:])
        - torch.maximum(local_boxes[:, :, 0, None], edges[:-1])
    ).clamp_min(0)
    vertical_coverage = (
        torch.minimum(local_boxes[:, :, 3, None], edges[1:])
        - torch.maximum(local_boxes[:, :, 1, None], edges[:-1])
    ).clamp_min(0)
    weights = (
        vertical_coverage.unsqueeze(-1) * horizontal_coverage.unsqueeze(-2)
    ).flatten(start_dim=2)
    weights = weights * valid[:, None, None]
    return weights, valid, intersection_area


class RegionLoss(nn.Module):
    def __init__(self, min_area):
        super().__init__()
        if not 0.0 <= min_area <= 1.0:
            raise ValueError("region_min_area must be between 0 and 1")
        self.min_area = min_area

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_patch_logits, crop_boxes, teacher_temp):
        """Select raw overlapping patch logits, apply SK, then pool by area.

        Both views and all ranks share one assignment problem. Invalid
        intersections and patches with zero coverage never participate.
        Every rank must call this, even if it has no valid local overlap.
        """
        weights, valid, intersection_area = intersection_patch_weights(
            crop_boxes.float(), teacher_patch_logits.shape[1], self.min_area
        )
        batch_size = len(crop_boxes)
        if teacher_patch_logits.shape[0] != 2 * batch_size:
            raise ValueError("Overlap teacher logits must contain two global views")

        # Model output is view-major: all samples of view 0, then view 1.
        view_weights = weights.transpose(0, 1).reshape(teacher_patch_logits.shape[:2])
        selected = view_weights > 0
        assignments = sinkhorn_knopp(teacher_patch_logits[selected], teacher_temp)

        normalized_weights = view_weights / view_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        # Reduce only selected patches instead of allocating another dense
        # [2 * batch, patches, prototypes] probability tensor.
        region_indices = selected.nonzero(as_tuple=True)[0]
        assignments *= normalized_weights[selected].unsqueeze(-1)
        pooled = assignments.new_zeros((2 * batch_size, teacher_patch_logits.shape[-1]))
        pooled.index_add_(0, region_indices, assignments)
        pooled = pooled.reshape(2, batch_size, -1).transpose(0, 1)
        return OverlapTeacherTargets(pooled, weights, valid, intersection_area)

    def forward(
        self,
        student_patch_logits,
        teacher_patch_probabilities,
        crop_boxes,
        *,
        teacher_overlap_targets=None,
    ):
        if len(student_patch_logits) != 2 or len(teacher_patch_probabilities) != 2:
            raise ValueError("Region loss requires exactly two global crops")

        if teacher_overlap_targets is None:
            weights, valid, intersection_area = intersection_patch_weights(
                crop_boxes.float(),
                student_patch_logits[0].shape[1],
                self.min_area,
            )
        else:
            weights = teacher_overlap_targets.patch_weights
            valid = teacher_overlap_targets.valid
            intersection_area = teacher_overlap_targets.intersection_area
        all_patch_weights = weights

        valid_count = valid.sum().float()
        local_valid_samples = int(valid_count.item())
        global_valid_count = valid_count.detach().clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_valid_count)
            world_size = dist.get_world_size()

        if local_valid_samples:
            selector = (
                slice(None)
                if local_valid_samples == valid.numel()
                else valid
            )
            weights = weights[selector]
            normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            student_regions = []
            teacher_regions = []
            for view in range(2):
                student_probabilities = F.softmax(
                    student_patch_logits[view][selector].float(), dim=-1
                )
                student_regions.append(
                    torch.bmm(
                        weights[:, view].unsqueeze(1), student_probabilities
                    ).squeeze(1)
                    / normalizer[:, view]
                )
                if teacher_overlap_targets is None:
                    teacher_regions.append(
                        torch.bmm(
                            weights[:, view].unsqueeze(1),
                            teacher_patch_probabilities[view][selector].float(),
                        ).squeeze(1)
                        / normalizer[:, view]
                    )
                else:
                    teacher_regions.append(
                        teacher_overlap_targets.probabilities[selector, view].detach()
                    )
            student_region = torch.stack(student_regions, dim=1)
            teacher_region = torch.stack(teacher_regions, dim=1)

            loss_ab = -(
                teacher_region[:, 0]
                * student_region[:, 1].clamp_min(1e-12).log()
            ).sum(dim=-1)
            loss_ba = -(
                teacher_region[:, 1]
                * student_region[:, 0].clamp_min(1e-12).log()
            ).sum(dim=-1)
            local_loss_sum = (0.5 * (loss_ab + loss_ba)).sum()
        else:
            local_loss_sum = sum(
                logits.sum() * 0.0 for logits in student_patch_logits
            )

        loss = local_loss_sum * world_size / global_valid_count.clamp_min(1.0)

        return {
            "loss": loss,
            "valid_ratio": valid.float().mean(),
            "intersection_area": intersection_area.mean(),
            # Reuse the already-computed geometry for diagnostics in iBOTLoss.
            # These tensors are removed before the scalar logging dictionary is
            # returned to the training loop.
            "patch_weights": all_patch_weights,
            "valid": valid,
        }
