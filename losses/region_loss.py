import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def intersection_patch_fractions(crop_boxes, patch_count, min_area):
    """Return the fraction of each patch covered by the two-view intersection.

    Crop boxes use normalized original-image coordinates in the form
    ``[left, top, right, bottom, horizontally_flipped]``.
    """
    if crop_boxes.ndim != 3 or crop_boxes.shape[1:] != (2, 5):
        raise ValueError(
            "crop_boxes must have shape [batch_size, 2, 5], got "
            f"{tuple(crop_boxes.shape)}"
        )

    grid_size = math.isqrt(patch_count)
    if patch_count <= 0 or grid_size * grid_size != patch_count:
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

    # Patch coordinates keep fully covered cells exactly at fraction 1,
    # including grids such as 14x14 where normalized edges round in float32.
    local_boxes = local_boxes * grid_size
    edges = torch.arange(
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
    fractions = (
        vertical_coverage.unsqueeze(-1) * horizontal_coverage.unsqueeze(-2)
    ).flatten(start_dim=2)
    fractions = fractions.clamp(0, 1) * valid[:, None, None]
    return fractions, valid, intersection_area


class RegionLoss(nn.Module):
    """Symmetric CE between means of selected, uncentered patch softmaxes."""

    def __init__(self, min_area=0.0, patch_threshold=0.5, temperature=0.1):
        super().__init__()
        if not 0.0 <= min_area <= 1.0:
            raise ValueError("region_min_area must be between 0 and 1")
        if not 0.0 < patch_threshold <= 1.0:
            raise ValueError("region_patch_threshold must be in (0, 1]")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("region_temp must be finite and positive")
        self.min_area = min_area
        self.patch_threshold = patch_threshold
        self.temperature = temperature

    def _region_log_distribution(self, logits, selected):
        # log(mean(softmax(z / T))) preserves the exact probability-mean
        # objective without clamping away gradients for small probabilities.
        # Excluded logits never enter softmax or the region representation.
        logits = logits.float().masked_fill(~selected[..., None], 0)
        log_probabilities = F.log_softmax(logits / self.temperature, dim=-1)
        log_probabilities = log_probabilities.masked_fill(~selected[..., None], -torch.inf)
        return torch.logsumexp(log_probabilities, dim=1) - selected.sum(
            dim=1, keepdim=True
        ).float().log()

    def forward(self, student_patch_logits, teacher_patch_logits, crop_boxes):
        if len(student_patch_logits) != 2 or len(teacher_patch_logits) != 2:
            raise ValueError("Region loss requires exactly two global crops")
        shape = student_patch_logits[0].shape
        if (len(shape) != 3 or shape[0] != len(crop_boxes)
                or any(logits.shape != shape for logits in
                       (*student_patch_logits, *teacher_patch_logits))):
            raise ValueError("Region logits must have matching [batch, patches, prototypes] shapes")
        fractions, valid, intersection_area = intersection_patch_fractions(
            crop_boxes.float(), shape[1], self.min_area
        )
        selected = (fractions >= self.patch_threshold) & valid[:, None, None]
        # Both cross-view directions require at least one patch in each view.
        valid = valid & selected.any(dim=-1).all(dim=-1)
        selected = selected & valid[:, None, None]
        global_valid_count = valid.sum().float()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_valid_count)
            world_size = dist.get_world_size()

        if valid.any():
            student_regions = [
                self._region_log_distribution(logits[valid], selected[valid, view])
                for view, logits in enumerate(student_patch_logits)
            ]
            with torch.no_grad():
                teacher_regions = [
                    self._region_log_distribution(logits.detach()[valid], selected[valid, view]).exp()
                    for view, logits in enumerate(teacher_patch_logits)
                ]
            loss_ab = -(teacher_regions[0] * student_regions[1]).sum(dim=-1)
            loss_ba = -(teacher_regions[1] * student_regions[0]).sum(dim=-1)
            local_loss_sum = (0.5 * (loss_ab + loss_ba)).sum()
        else:
            # Empty ranks must still participate in DDP backward with zero grads.
            local_loss_sum = sum(logits.float().sum() * 0.0 for logits in student_patch_logits)

        loss = local_loss_sum * world_size / global_valid_count.clamp_min(1.0)
        return {
            "loss": loss,
            "valid_ratio": valid.float().mean(),
            "intersection_area": intersection_area.mean(),
            "patch_mask": selected,
            "valid": valid,
        }
