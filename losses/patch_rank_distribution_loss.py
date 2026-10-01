"""Regional histograms of patch-wise neighborhood rankings (no trainable head)."""

from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .region_ordering_loss import (
    collect_overlap_regions, gather_reference_bank, sample_external_references,
    SORT_QUERY_CHUNK, SORT_PERMUTATION_ELEMENTS,
)
from .region_sorting import bitonic_permutation

# NeCo's 7x7 reference grid has 49 identities. Bound the quadratic patch sorter
# independently of world size; use every available external region if fewer.
MAX_REFERENCE_REGIONS = 49


class _MeanPatchPermutations(torch.autograd.Function):
    """Tile exact permutation means; recompute only student sorting in backward.

    Saves scalar cosines/region indices, not per-patch MxM matrices or sorting
    stages. Supports first-order training gradients, like regional softmax pooling.
    """

    @staticmethod
    def forward(ctx, similarities, region_indices, region_count, tile):
        size = similarities.shape[1]
        counts = torch.bincount(region_indices, minlength=region_count).clamp_min(1)
        pooled = similarities.new_zeros(region_count, size, size)
        for start in range(0, len(similarities), tile):
            _, permutations = bitonic_permutation(similarities[start:start + tile])
            pooled.index_add_(0, region_indices[start:start + tile], permutations)
        ctx.save_for_backward(similarities, region_indices, counts)
        ctx.tile = tile
        return pooled / counts[:, None, None]

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        similarities, region_indices, counts = ctx.saved_tensors
        result = torch.empty_like(similarities)
        for start in range(0, len(similarities), ctx.tile):
            stop = start + ctx.tile
            with torch.enable_grad():
                values = similarities[start:stop].detach().requires_grad_()
                _, permutations = bitonic_permutation(values)
                rows = region_indices[start:stop]
                upstream = gradient.index_select(0, rows) / counts[rows, None, None]
                result[start:stop] = torch.autograd.grad(permutations, values, upstream)[0]
        return result, None, None, None


def mean_patch_permutations(similarities, region_indices, region_count):
    """Average full identity-by-rank matrices, allowing unequal patch counts."""
    size = similarities.shape[1]
    tile = min(SORT_QUERY_CHUNK, max(1, SORT_PERMUTATION_ELEMENTS // (size * size)))
    return _MeanPatchPermutations.apply(similarities.float(), region_indices, region_count, tile)


def _regional_rank_matrices(views, masks, images, bank, reference_indices):
    """Shared GEMMs and scalar gathers; no [patches, references, features] copy."""
    pooled = []
    for view, mask in zip(views, masks):
        selected = mask[images, 0]
        region_rows, patch_rows = selected.nonzero(as_tuple=True)
        patches = view[images[region_rows], patch_rows].float()
        cosines = F.normalize(patches, dim=-1) @ bank.T
        cosines = cosines.gather(1, reference_indices.index_select(0, region_rows))
        pooled.append(mean_patch_permutations(cosines, region_rows, len(images)))
    return pooled


class PatchRankDistributionLoss(nn.Module):
    """Sort continuous patches, mean matrices within overlap, cross-view CE."""

    def __init__(self, seed=0, min_area=.1):
        super().__init__()
        self.seed = int(seed)
        self.min_area = min_area
        self.register_buffer("sampling_step", torch.zeros((), dtype=torch.long))

    def forward(self, student_views, teacher_views, crop_boxes):
        with torch.autocast(device_type=student_views[0].device.type, enabled=False):
            return self._forward(student_views, teacher_views, crop_boxes)

    def _forward(self, student_views, teacher_views, crop_boxes):
        if len(student_views) != 2 or len(teacher_views) != 2:
            raise ValueError("patch_rank_distribution requires two global backbone feature views")
        batch, patches, dimensions = student_views[0].shape
        if any(view.shape != (batch, patches, dimensions) for view in (*student_views, *teacher_views)):
            raise ValueError("Teacher/student backbone patch feature shapes must match")
        regions, masks = collect_overlap_regions(
            crop_boxes[:, :2], [patches, patches], global_only=True, min_area=self.min_area,
        )
        device = student_views[0].device
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        generator = torch.Generator().manual_seed(self.seed + 1000003 * int(self.sampling_step.item()) + rank)
        self.sampling_step.add_(1)
        eligible = [image for image, entries in enumerate(regions) if entries]
        images = torch.tensor(eligible, device=device, dtype=torch.long)
        with torch.no_grad():
            # Normalize every patch, mean only contained patches, normalize mean.
            references = torch.zeros(len(images), dimensions, device=device)
            for view in range(2):
                weights = masks[view][images, 0].float()
                normalized = F.normalize(teacher_views[view].detach().index_select(0, images).float(), dim=-1)
                means = (weights[:, None] @ normalized).squeeze(1) / weights.sum(1, keepdim=True).clamp_min(1)
                use_view = torch.tensor([regions[image][0].teacher_view == view for image in eligible],
                                        device=device, dtype=torch.bool)
                references = torch.where(use_view[:, None], means, references)
            references = F.normalize(references, dim=-1)
            owners = torch.tensor([(rank, image) for image in eligible], device=device, dtype=torch.long).reshape(-1, 2)
            bank, bank_owners = gather_reference_bank(references, owners)
        owner_rows = defaultdict(list)
        for index, owner in enumerate(bank_owners.cpu().tolist()):
            owner_rows[tuple(owner)].append(index)
        references_per_query = min(MAX_REFERENCE_REGIONS, max(0, len(bank) - 1))
        zero = sum(view.reshape(-1)[:1].float().sum() * 0 for view in student_views)
        if eligible and references_per_query >= 2:
            reference_indices = torch.tensor([
                sample_external_references(owner_rows, (rank, image), references_per_query, generator)
                for image in eligible
            ], device=device, dtype=torch.long)
            # Both directions use the same identities and input order. Teacher
            # matrices are computed once, detached, and never recomputed backward.
            with torch.no_grad():
                teacher_matrices = _regional_rank_matrices(teacher_views, masks, images, bank, reference_indices)
            student_matrices = _regional_rank_matrices(student_views, masks, images, bank, reference_indices)
            losses = [-(teacher_matrices[t] * student_matrices[s].clamp_min(1e-12).log()).sum(1).mean(1)
                      for t, s in ((0, 1), (1, 0))]
            loss_sum = sum(value.sum() for value in losses) + zero
            query_count = 2 * len(eligible)
        else:
            loss_sum, query_count = zero, 0
        counts = zero.new_tensor([query_count, query_count * references_per_query, len(eligible)])
        world = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            world = dist.get_world_size()
        # DDP averages gradients, so scale local sum by world/global query count.
        loss = loss_sum * world / counts[0].clamp_min(1)
        return {"loss": loss, "query_count": counts[0],
                "references_per_query": counts[1] / counts[0].clamp_min(1),
                "reference_region_count": counts[2]}
