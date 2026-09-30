"""Auxiliary ordering of overlap compositions using NeCo permutation matching."""

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .region_loss import intersection_patch_fractions
from .region_sorting import permutation_cross_entropy


ORDERING_WEIGHT = .1
MIN_REGION_PATCHES = 4
FAMILY_NAMES = ("global_global", "global_local", "local_local")
FAMILY_WEIGHTS = (.5, .25, .25)


def validate_loss_modality(modality, aggregation, normalization, threshold, ngcrops, nlcrops):
    if modality not in ("standard", "cross_image", "within_image"):
        raise ValueError("loss_modality must be standard, cross_image, or within_image")
    if modality != "standard" and (
        aggregation != "mean" or normalization != "centering" or threshold != 1.0
        or ngcrops != 2 or (modality == "within_image" and nlcrops < 2)
    ):
        raise ValueError(
            "Ordering loss_modality requires mean aggregation, centering, "
            "region_patch_threshold: 1.0, two globals and (for within_image) at least two locals"
        )


@dataclass
class OverlapRegion:
    coordinates: tuple
    row: int
    teacher_view: int
    # (teacher global view, student view) -> family; duplicate physical
    # overlaps/directions are counted once, with GG > GL > LL precedence.
    queries: dict = field(default_factory=dict)


def collect_overlap_regions(crop_boxes, patch_counts, *, global_only=False, min_area=0.0):
    """All crop combinations, deduplicated in original-image coordinates.

    Compute native-grid masks in a batch on the input device, including flips.
    Move only small geometry/count tables to CPU for region deduplication.
    A local/local query uses the best covering global teacher, never a local
    teacher pass. A reference always uses the covering global with most fully
    contained patches (ties choose the first global).
    """
    batch, views, _ = crop_boxes.shape
    if crop_boxes.shape[2] != 5 or views != len(patch_counts) or views < (2 if global_only else 4):
        raise ValueError("Ordering requires geometry and patch counts for all global/local views")
    boxes = crop_boxes.float()
    pairs = [(0, 1)] if global_only else list(combinations(range(views), 2))
    first, second = zip(*pairs)
    coordinates = torch.cat((
        torch.maximum(boxes[:, first, :2], boxes[:, second, :2]),
        torch.minimum(boxes[:, first, 2:4], boxes[:, second, 2:4]),
    ), -1)
    positive = (coordinates[..., 2:] > coordinates[..., :2]).all(-1)
    positive &= (coordinates[..., 2:] - coordinates[..., :2]).clamp_min(0).prod(-1) >= min_area
    regions = torch.cat((coordinates, torch.zeros_like(coordinates[..., :1])), -1)
    masks = []
    for view, patches in enumerate(patch_counts):
        paired = torch.stack((boxes[:, view, None].expand_as(regions), regions), 2)
        fractions, _, _ = intersection_patch_fractions(paired.reshape(-1, 2, 5), patches, 0.0)
        masks.append((fractions[:, 0] >= 1.0).reshape(batch, len(pairs), patches))
    counts = torch.stack([mask.sum(-1) for mask in masks], -1)
    covering = torch.stack([
        (boxes[:, view, None, :2] <= coordinates[..., :2]).all(-1)
        & (boxes[:, view, None, 2:4] >= coordinates[..., 2:]).all(-1)
        & (counts[..., view] >= MIN_REGION_PATCHES) & positive
        for view in range(2)
    ], -1)
    coordinates_cpu = coordinates.detach().cpu().tolist()
    counts_cpu = counts.cpu().tolist()
    covering_cpu = covering.cpu().tolist()
    by_image = []
    for image in range(batch):
        distinct = {}
        for row, (a, b) in enumerate(pairs):
            candidates = [view for view in range(2) if covering_cpu[image][row][view]]
            if not candidates or min(counts_cpu[image][row][a], counts_cpu[image][row][b]) < MIN_REGION_PATCHES:
                continue
            teacher = max(candidates, key=lambda view: (counts_cpu[image][row][view], -view))
            key = tuple(coordinates_cpu[image][row])
            region = distinct.setdefault(key, OverlapRegion(key, row, teacher))
            if b < 2:
                directions = ((0, 1), (1, 0))
                family = 0
            elif a < 2:
                # Preserve the paired global's teacher context. The best-
                # covering-global rule is for reference identity, not a reason
                # to merge queries from two different global teacher views.
                directions = ((a, b),)
                family = 1
            else:
                directions = ((teacher, a), (teacher, b))
                family = 2
            for teacher_view, student_view in directions:
                previous = region.queries.get((teacher_view, student_view), family)
                region.queries[teacher_view, student_view] = min(previous, family)
        by_image.append(list(distinct.values()))
    return by_image, masks


@torch.no_grad()
def gather_reference_bank(local_bank, local_owners, gather_vectors=True):
    """Variable-size detached banks, including ranks with no eligible regions.

    Owners are (rank, minibatch image index); only cross_image needs to transmit
    prototype vectors. Within-image ordering does not need a distributed bank.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_bank.detach(), local_owners
    world = dist.get_world_size()
    count = torch.tensor([len(local_bank)], device=local_bank.device, dtype=torch.long)
    counts = [torch.zeros_like(count) for _ in range(world)]
    dist.all_gather(counts, count)
    sizes = [int(value.item()) for value in counts]
    maximum = max(sizes)
    if maximum == 0:
        return local_bank.detach(), local_owners
    def gather(tensor):
        padded = tensor.new_zeros((maximum,) + tensor.shape[1:])
        padded[:len(tensor)] = tensor
        outputs = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(outputs, padded)
        return torch.cat([value[:size] for value, size in zip(outputs, sizes)])
    owners = gather(local_owners)
    bank = gather(local_bank.detach()) if gather_vectors else local_bank.detach()
    return bank, owners


def sample_external_references(owner_rows, query_owner, count, generator):
    """Random round-robin across shuffled images, without region replacement."""
    images = [owner for owner in owner_rows if owner != query_owner]
    if sum(len(owner_rows[owner]) for owner in images) < count:
        return None
    images = [images[index] for index in torch.randperm(len(images), generator=generator).tolist()]
    pools = {
        owner: [owner_rows[owner][index] for index in torch.randperm(
            len(owner_rows[owner]), generator=generator
        ).tolist()] for owner in images
    }
    chosen = []
    while len(chosen) < count:
        for owner in images:
            if pools[owner]:
                chosen.append(pools[owner].pop())
                if len(chosen) == count:
                    break
    return chosen


def _pool_student(logits, weights, temperature):
    probabilities = F.softmax(logits.float() / temperature, -1)
    return weights.float() @ probabilities / weights.sum(-1, keepdim=True)


def _compare_queries(students, teachers, bank, indices):
    # Gather inside the checkpoint: retain the shared bank and small indices,
    # rather than a separate M x D bank for every query's backward pass.
    references = bank[indices]
    student_cosines = torch.einsum("qd,qmd->qm", F.normalize(students, dim=-1), references)
    with torch.no_grad():
        teacher_cosines = torch.einsum("qd,qmd->qm", F.normalize(teachers.detach(), dim=-1), references)
    return permutation_cross_entropy(student_cosines, teacher_cosines)


class RegionOrderingLoss(nn.Module):
    def __init__(self, modality, student_temperature=.1, seed=0, min_area=0.0):
        super().__init__()
        self.modality = modality
        self.student_temperature = student_temperature
        self.seed = int(seed)
        self.min_area = min_area
        self.register_buffer("sampling_step", torch.zeros((), dtype=torch.long))
        if modality == "cross_image":
            self.register_buffer("single_global_overlap", torch.tensor(True))

    def forward(self, student_views, teacher_targets, crop_boxes):
        # Keep cosine and sorter calculations in float32 under FP16/BF16 training.
        with torch.autocast(device_type=student_views[0].device.type, enabled=False):
            return self._forward(student_views, teacher_targets, crop_boxes)

    def _forward(self, student_views, teacher_targets, crop_boxes):
        if self.modality == "cross_image":
            # Exactly one physical region per image: the same two-global
            # intersection as standard. Local patches/geometry never enter.
            student_views = student_views[:2]
            crop_boxes = crop_boxes[:, :2]
        batch, _, dimensions = student_views[0].shape
        if len(teacher_targets) != 2 or crop_boxes.shape != (batch, len(student_views), 5):
            raise ValueError("Ordering requires two global teacher outputs and every student crop box")
        if any(view.ndim != 3 or view.shape[0] != batch or view.shape[2] != dimensions
               for view in (*student_views, *teacher_targets)):
            raise ValueError("Ordering patch distributions must have matching batch/prototype dimensions")
        if any(teacher.shape[1] != student.shape[1]
               for teacher, student in zip(teacher_targets, student_views[:2])):
            raise ValueError("Teacher/student global patch grids must match")
        regions, masks = collect_overlap_regions(
            crop_boxes, [view.shape[1] for view in student_views],
            global_only=self.modality == "cross_image",
            min_area=self.min_area if self.modality == "cross_image" else 0.0,
        )
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        # Separate RNG leaves augmentations/masking identical across ablations.
        generator = torch.Generator().manual_seed(self.seed + 1000003 * int(self.sampling_step.item()) + rank)
        self.sampling_step.add_(1)

        teacher_pooled = {}
        reference_rows, owners = [], []
        with torch.no_grad():
            for image, image_regions in enumerate(regions):
                for view in range(2):
                    needed = [index for index, region in enumerate(image_regions)
                              if region.teacher_view == view or any(t == view for t, _ in region.queries)]
                    if not needed:
                        continue
                    weights = masks[view][image, [image_regions[index].row for index in needed]].float()
                    pooled = weights @ teacher_targets[view][image].detach().float() / weights.sum(-1, keepdim=True)
                    for index, value in zip(needed, pooled):
                        teacher_pooled[image, index, view] = value
                for index, region in enumerate(image_regions):
                    reference_rows.append(teacher_pooled[image, index, region.teacher_view])
                    owners.append((rank, image))
        local_bank = (torch.stack(reference_rows) if reference_rows else
                      student_views[0].new_empty((0, dimensions), dtype=torch.float32))
        local_owners = torch.tensor(owners, device=local_bank.device, dtype=torch.long).reshape(-1, 2)
        bank, bank_owners = (gather_reference_bank(local_bank, local_owners)
                             if self.modality == "cross_image" else (local_bank, local_owners))
        owner_rows = defaultdict(list)
        for index, owner in enumerate(bank_owners.cpu().tolist()):
            owner_rows[tuple(owner)].append(index)
        bank = F.normalize(bank.detach(), dim=-1)
        local_offsets = []
        offset = 0
        for image_regions in regions:
            local_offsets.append(offset)
            offset += len(image_regions)

        # Cross-image references are the single global/global regions of all
        # other eligible images. Within-image references are the other distinct
        # physical regions of this image; it never depends on external images.
        groups = defaultdict(list)
        reference_count = 0
        for image, image_regions in enumerate(regions):
            count = (sum(len(rows) for owner, rows in owner_rows.items() if owner != (rank, image))
                     if self.modality == "cross_image" else len(image_regions) - 1)
            if count < 2:
                continue
            for index, region in enumerate(image_regions):
                if self.modality == "cross_image":
                    references = sample_external_references(owner_rows, (rank, image), count, generator)
                else:
                    references = [local_offsets[image] + other
                                  for other in range(len(image_regions)) if other != index]
                # Teacher and student use the exact same identities/input order.
                for (teacher_view, student_view), family in region.queries.items():
                    groups[count].append((image, index, teacher_view, student_view, family, references))
                    reference_count += count

        zero = sum(view.sum(dtype=torch.float32) * 0.0 for view in student_views)
        family_sums = [zero for _ in FAMILY_NAMES]
        family_counts = [0 for _ in FAMILY_NAMES]
        # Pool each required view/physical region once. Activation checkpointing
        # avoids keeping all local patch softmaxes and sorter stages in memory.
        student_pooled = {}
        needed_views = defaultdict(set)
        for queries in groups.values():
            for image, index, _, view, _, _ in queries:
                needed_views[image, view].add(index)
        for (image, view), needed in needed_views.items():
            needed = sorted(needed)
            weights = masks[view][image, [regions[image][index].row for index in needed]].float()
            pooled = checkpoint(_pool_student, student_views[view][image], weights,
                                self.student_temperature, use_reentrant=False)
            for index, value in zip(needed, pooled):
                student_pooled[image, index, view] = value

        for queries in groups.values():
            for start in range(0, len(queries), 32):
                chunk = queries[start:start + 32]
                indices = torch.tensor([query[-1] for query in chunk], device=bank.device, dtype=torch.long)
                students = torch.stack([student_pooled[q[0], q[1], q[3]] for q in chunk])
                teachers = torch.stack([teacher_pooled[q[0], q[1], q[2]] for q in chunk])
                losses = checkpoint(_compare_queries, students, teachers, bank, indices, use_reentrant=False)
                for family in range(3):
                    positions = [index for index, query in enumerate(chunk) if query[4] == family]
                    if positions:
                        family_sums[family] = family_sums[family] + losses[positions].sum()
                        family_counts[family] += len(positions)

        counts = zero.new_tensor([*family_counts, reference_count, len(reference_rows)])
        world = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            world = dist.get_world_size()
        family_means = torch.stack(family_sums) * world / counts[:3].clamp_min(1)
        weights = counts.new_tensor(FAMILY_WEIGHTS) * (counts[:3] > 0)
        loss = (family_means * weights).sum() / weights.sum().clamp_min(.25)
        return {
            "loss": loss,
            "query_count": counts[:3].sum(),
            "references_per_query": counts[3] / counts[:3].sum().clamp_min(1),
            "reference_region_count": counts[4],
            **{f"{name}_loss": value for name, value in zip(FAMILY_NAMES, family_means)},
            **{f"{name}_queries": value for name, value in zip(FAMILY_NAMES, counts[:3])},
        }
