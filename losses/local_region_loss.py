"""Per-image global/global and global/local regional composition ablation."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .region_loss import intersection_patch_fractions
from .sinkhorn import sinkhorn_log_probabilities


def validate_local_region_settings(enabled, aggregation, normalization):
    if type(enabled) is not bool:
        raise ValueError("include_local_crops must be a boolean")
    if enabled and (aggregation != "mean" or normalization not in (
        "centering", "softmax", "sinkhorn",
    )):
        raise ValueError(
            "include_local_crops requires region_aggregation: mean and "
            "region_normalization: centering, softmax, or sinkhorn"
        )


def global_local_region_loss(
    region_loss, global_stats, student_local_logits, teacher_global_logits,
    crop_boxes, *, teacher_patch_targets=None,
):
    """Teacher global -> student local CE; no local teacher forward.

    Global/global keeps its existing area/selection rules. Global/local uses
    positive intersections and fully contained patches on each native grid.
    Average pairs within each image, then mix groups and average valid images
    across ranks. The fixed group weights are renormalized for missing groups.
    """
    if len(teacher_global_logits) != 2 or not student_local_logits:
        raise ValueError("Global/local region loss requires two globals and local patch logits")
    batch, global_patches, prototypes = teacher_global_logits[0].shape
    if crop_boxes.shape != (batch, 2 + len(student_local_logits), 5):
        raise ValueError("include_local_crops requires crop boxes for every global and local view")
    if any(x.ndim != 3 or x.shape[0] != batch or x.shape[2] != prototypes
           for x in student_local_logits):
        raise ValueError("Local patch logits must match global batch and prototype dimensions")

    # Geometry is independent of normalization. Exclude a pair unless both
    # views contain at least one complete patch. Horizontal flips are retained.
    pairs = []
    teacher_selected = [torch.zeros(
        batch, global_patches, dtype=torch.bool, device=crop_boxes.device
    ) for _ in range(2)]
    pair_counts = crop_boxes.new_zeros(batch)
    for global_view in range(2):
        for local_view, logits in enumerate(student_local_logits):
            boxes = crop_boxes[:, [global_view, 2 + local_view]].float()
            fractions, positive, _ = intersection_patch_fractions(
                boxes, (global_patches, logits.shape[1]), min_area=0.0
            )
            weights = tuple((fraction >= 1.0).float() for fraction in fractions)
            valid = positive & (weights[0].sum(1) > 0) & (weights[1].sum(1) > 0)
            pairs.append((global_view, local_view, weights, valid))
            teacher_selected[global_view] |= weights[0].bool() & valid[:, None]
            pair_counts += valid.float()

    # Transform each view once and reuse it for all its intersections. Centered
    # teacher targets are the very same probabilities used by ordinary iBOT.
    student_logs = tuple(F.log_softmax(x.float() / region_loss.student_temperature, -1)
                         for x in student_local_logits)
    with torch.no_grad():
        if region_loss.normalization == "centering":
            if teacher_patch_targets is None:
                raise ValueError("Centered global/local loss requires teacher patch targets")
            teacher_probabilities = tuple(x.detach().float() for x in teacher_patch_targets)
        elif region_loss.normalization == "softmax":
            teacher_probabilities = tuple(
                F.softmax(x.detach().float() / region_loss.temperature, -1)
                for x in teacher_global_logits
            )
        else:
            # Unique participating teacher patches across all valid local
            # intersections, jointly balanced across views/ranks, then reused.
            selected = torch.stack(teacher_selected, 1)
            logits = torch.stack(teacher_global_logits, 1).detach()
            assignments = sinkhorn_log_probabilities(logits[selected], region_loss.temperature).exp()
            dense = logits.new_zeros(logits.shape, dtype=torch.float32)
            dense[selected] = assignments
            teacher_probabilities = dense.unbind(1)

    local_sums = sum(x.float().sum((1, 2)) * 0.0 for x in student_local_logits)
    for global_view, local_view, weights, valid in pairs:
        if not valid.any():
            continue
        with torch.no_grad():
            target = region_loss._region_probability_mean(
                teacher_probabilities[global_view][valid], weights[0][valid]
            )
        prediction = region_loss._pool_log_probabilities(
            student_logs[local_view][valid], weights[1][valid]
        )
        cross_entropy = -(target * prediction).sum(-1)
        local_sums = local_sums.index_add(0, valid.nonzero().flatten(), cross_entropy)

    global_valid = global_stats["valid"]
    local_valid = pair_counts > 0
    valid = global_valid | local_valid
    local_means = local_sums / pair_counts.clamp_min(1)
    global_means = global_stats["per_image_loss"]
    denominator = .75 * global_valid.float() + .25 * local_valid.float()
    combined = (.75 * global_means + .25 * local_means) / denominator.clamp_min(.25)

    counts = torch.stack([valid.sum(), global_valid.sum(), local_valid.sum()]).float()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
        world_size = dist.get_world_size()
    return {
        **global_stats,
        "loss": combined.sum() * world_size / counts[0].clamp_min(1),
        "valid_ratio": valid.float().mean(),
        "global_global_loss": global_means.sum() * world_size / counts[1].clamp_min(1),
        "global_local_loss": local_means.sum() * world_size / counts[2].clamp_min(1),
        "global_local_valid_ratio": local_valid.float().mean(),
        "global_local_pairs_per_image": pair_counts.mean(),
    }
