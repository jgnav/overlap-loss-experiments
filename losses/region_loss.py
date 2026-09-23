import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .sinkhorn import sinkhorn_log_probabilities
from .region_aggregation import RegionAggregation


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
    """Symmetric consistency with binary or overlap-area-weighted pooling."""

    def __init__(self, min_area=0.0, patch_threshold=0.5, temperature=0.1,
                 normalization="softmax", student_temperature=0.1,
                 aggregation="mean"):
        super().__init__()
        if not 0.0 <= min_area <= 1.0:
            raise ValueError("region_min_area must be between 0 and 1")
        if patch_threshold != "weighted" and not (
            type(patch_threshold) in (int, float) and 0.0 < patch_threshold <= 1.0
        ):
            raise ValueError("region_patch_threshold must be in (0, 1] or 'weighted'")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("region_temp must be finite and positive")
        if not math.isfinite(student_temperature) or student_temperature <= 0:
            raise ValueError("student_temperature must be finite and positive")
        if normalization not in ("centering", "softmax", "raw_logits", "sinkhorn"):
            raise ValueError(
                "region_normalization must be centering, softmax, raw_logits, or sinkhorn"
            )
        self.min_area = min_area
        self.patch_threshold = patch_threshold
        self.temperature = temperature
        self.student_temperature = student_temperature
        self.normalization = normalization
        self.aggregation = RegionAggregation(aggregation)
        if aggregation == "hellinger" and normalization == "raw_logits":
            raise ValueError("hellinger aggregation requires probability distributions, not raw_logits")

    @staticmethod
    def _region_raw_vector(logits, weights):
        selected = weights > 0
        logits = logits.float().masked_fill(~selected[..., None], 0)
        patches = F.normalize(logits, p=2, dim=-1)
        return (patches * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True)

    def _sinkhorn_patches(self, logits, selected):
        # One assignment problem over both views and all valid selected patches,
        # independently for student and teacher. Keep the student graph intact.
        logits = torch.stack(logits, dim=1)
        assignments = sinkhorn_log_probabilities(logits[selected], self.temperature)
        dense = logits.new_full(logits.shape, -torch.inf, dtype=torch.float32)
        return dense.masked_scatter(selected[..., None], assignments), assignments

    def _pool_log_probabilities(self, log_probabilities, weights):
        if self.aggregation.method == "hellinger":
            # r=1/2: square the weighted mean of square roots, then
            # normalize across prototypes. Stay in log space for gradients.
            pooled = 2 * torch.logsumexp(
                .5 * log_probabilities + weights.float().log()[..., None], dim=1
            )
            return F.log_softmax(pooled, dim=-1)
        # log(sum(w * p) / sum(w)); zero coverage contributes exactly zero.
        log_weights = weights.float().log()
        return torch.logsumexp(log_probabilities + log_weights[..., None], dim=1) - weights.sum(
            dim=1, keepdim=True
        ).float().log()

    def _region_log_distribution(self, logits, weights, temperature=None):
        # Log-space weighted pooling preserves the exact probability-mean
        # objective without clamping away gradients for small probabilities.
        # Excluded logits never enter softmax or the region representation.
        selected = weights > 0
        logits = logits.float().masked_fill(~selected[..., None], 0)
        temperature = self.temperature if temperature is None else temperature
        log_probabilities = F.log_softmax(logits / temperature, dim=-1)
        log_probabilities = log_probabilities.masked_fill(~selected[..., None], -torch.inf)
        return self._pool_log_probabilities(log_probabilities, weights)

    def _region_probability_mean(self, probabilities, weights):
        selected = weights > 0
        probabilities = probabilities.float().masked_fill(
            ~selected[..., None], 0
        )
        if self.aggregation.method == "hellinger":
            pooled = (probabilities.sqrt() * weights[..., None]).sum(dim=1).square()
            return pooled / pooled.sum(dim=-1, keepdim=True)
        return (probabilities * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True)

    def forward(
        self,
        student_patch_logits,
        teacher_patch_logits,
        crop_boxes,
        *,
        teacher_patch_targets=None,
    ):
        if len(student_patch_logits) != 2 or len(teacher_patch_logits) != 2:
            raise ValueError("Region loss requires exactly two global crops")
        shape = student_patch_logits[0].shape
        if (len(shape) != 3 or shape[0] != len(crop_boxes)
                or any(logits.shape != shape for logits in
                       (*student_patch_logits, *teacher_patch_logits))):
            raise ValueError("Region logits must have matching [batch, patches, prototypes] shapes")
        if self.normalization == "centering":
            if teacher_patch_targets is None:
                raise ValueError(
                    "centering region normalization requires teacher_patch_targets"
                )
            if len(teacher_patch_targets) != 2 or any(
                target.shape != shape for target in teacher_patch_targets
            ):
                raise ValueError(
                    "Centered teacher patch targets must match region-logit shapes"
                )
        fractions, valid, intersection_area = intersection_patch_fractions(
            crop_boxes.float(), shape[1], self.min_area
        )
        weights = (fractions if self.patch_threshold == "weighted" else
                   (fractions >= self.patch_threshold).float())
        selected = (weights > 0) & valid[:, None, None]
        # Both cross-view directions require at least one patch in each view.
        valid = valid & selected.any(dim=-1).all(dim=-1)
        selected = selected & valid[:, None, None]
        weights = weights * selected
        global_valid_count = valid.sum().float()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_valid_count)
            world_size = dist.get_world_size()

        student_assignments = None
        if self.normalization == "sinkhorn" and global_valid_count.item() > 0:
            student_patches, student_assignments = self._sinkhorn_patches(
                student_patch_logits, selected
            )
            with torch.no_grad():
                teacher_patches, _ = self._sinkhorn_patches(
                    tuple(x.detach() for x in teacher_patch_logits), selected
                )

        if valid.any():
            if self.normalization == "sinkhorn":
                student_regions = [
                    self._pool_log_probabilities(student_patches[valid, v], weights[valid, v])
                    for v in range(2)
                ]
                teacher_regions = [
                    self._pool_log_probabilities(teacher_patches[valid, v], weights[valid, v]).exp()
                    for v in range(2)
                ]
            elif self.normalization == "centering":
                student_regions = [
                    self._region_log_distribution(
                        x[valid], weights[valid, view], self.student_temperature
                    )
                    for view, x in enumerate(student_patch_logits)
                ]
                with torch.no_grad():
                    teacher_regions = [
                        self._region_probability_mean(
                            target.detach()[valid], weights[valid, view]
                        )
                        for view, target in enumerate(teacher_patch_targets)
                    ]
            else:
                transform = (self._region_raw_vector if self.normalization == "raw_logits"
                             else self._region_log_distribution)
                student_regions = [transform(x[valid], weights[valid, v])
                                   for v, x in enumerate(student_patch_logits)]
                with torch.no_grad():
                    teacher_regions = [transform(x.detach()[valid], weights[valid, v])
                                       for v, x in enumerate(teacher_patch_logits)]
                    if self.normalization == "softmax":
                        teacher_regions = [x.exp() for x in teacher_regions]
            if self.normalization == "raw_logits":
                loss_ab = 1 - F.cosine_similarity(teacher_regions[0], student_regions[1], dim=-1)
                loss_ba = 1 - F.cosine_similarity(teacher_regions[1], student_regions[0], dim=-1)
            else:
                loss_ab = -(teacher_regions[0] * student_regions[1]).sum(dim=-1)
                loss_ba = -(teacher_regions[1] * student_regions[0]).sum(dim=-1)
            if self.aggregation.method not in ("mean", "hellinger"):
                # Coverage weights precede all statistics/distribution matching.
                def patch_values(logits, view, teacher=False):
                    if self.normalization == "sinkhorn":
                        bank = teacher_patches if teacher else student_patches
                        return bank[valid, view].exp()
                    if teacher and self.normalization == "centering":
                        return teacher_patch_targets[view].detach()[valid].float()
                    x = logits.detach() if teacher else logits
                    x = x[valid].float().masked_fill(~selected[valid, view, :, None], 0)
                    if self.normalization == "raw_logits":
                        return F.normalize(x, dim=-1)
                    temp = self.student_temperature if self.normalization == "centering" else self.temperature
                    return (x / temp).softmax(-1)

                sp = [patch_values(x, v) for v, x in enumerate(student_patch_logits)]
                with torch.no_grad():
                    tp = [patch_values(x, v, True) for v, x in enumerate(teacher_patch_logits)]
                # Matmul/projections and squared moments must stay float32
                # even inside the training loop's mixed-precision context.
                with torch.autocast(device_type=sp[0].device.type, enabled=False):
                    loss_ab = self.aggregation(sp[1], tp[0], weights[valid, 1], weights[valid, 0], loss_ab)
                    loss_ba = self.aggregation(sp[0], tp[1], weights[valid, 0], weights[valid, 1], loss_ba)
            local_loss_sum = (0.5 * (loss_ab + loss_ba)).sum()
        else:
            # Empty ranks must still participate in DDP backward with zero grads.
            local_loss_sum = sum(logits.float().sum() * 0.0 for logits in student_patch_logits)
            if student_assignments is not None:
                # Empty ranks still join the differentiable SK all-reduces.
                local_loss_sum = local_loss_sum + student_assignments.sum() * 0.0

        loss = local_loss_sum * world_size / global_valid_count.clamp_min(1.0)
        return {
            "loss": loss,
            "valid_ratio": valid.float().mean(),
            "intersection_area": intersection_area.mean(),
            "patch_mask": selected,
            "patch_weights": weights,
            "valid": valid,
        }
