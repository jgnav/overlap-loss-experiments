"""Auxiliary ordering of overlap compositions using NeCo permutation matching."""

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .region_sorting import bitonic_permutation, permutation_target_cross_entropy
from .region_pooling import region_probability_mean


ORDERING_WEIGHT = .1
MIN_REGION_PATCHES = 4
FAMILY_NAMES = ("global_global", "global_local", "local_local")
FAMILY_WEIGHTS = (.5, .25, .25)
POOL_SOFTMAX_BYTES = 64 * 1024 * 1024
SORT_QUERY_CHUNK = 128
SORT_PERMUTATION_ELEMENTS = 4 * 1024 * 1024


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


def _contained_patch_masks(boxes, coordinates, patch_counts):
    """Group equal grids; compute only the binary coverage needed by ordering."""
    groups = defaultdict(list)
    for view, patches in enumerate(patch_counts):
        grid = math.isqrt(patches)
        if patches <= 0 or grid * grid != patches:
            raise ValueError(f"The number of patch tokens must be square, got {patches}")
        groups[grid].append(view)
    masks = [None] * len(patch_counts)
    for grid, views in groups.items():
        crops = boxes[:, views]
        width = (crops[..., 2] - crops[..., 0]).clamp_min(1e-12)[..., None]
        height = (crops[..., 3] - crops[..., 1]).clamp_min(1e-12)[..., None]
        left = (coordinates[:, None, :, 0] - crops[..., 0, None]) / width
        right = (coordinates[:, None, :, 2] - crops[..., 0, None]) / width
        top = (coordinates[:, None, :, 1] - crops[..., 1, None]) / height
        bottom = (coordinates[:, None, :, 3] - crops[..., 1, None]) / height
        flipped = crops[..., 4, None] >= .5
        left, right = (torch.where(flipped, 1 - right, left),
                       torch.where(flipped, 1 - left, right))
        left, right, top, bottom = [value.clamp(0, 1) * grid for value in (left, right, top, bottom)]
        edges = torch.arange(grid + 1, device=boxes.device, dtype=boxes.dtype)
        horizontal = (torch.minimum(right[..., None], edges[1:]) -
                      torch.maximum(left[..., None], edges[:-1])).clamp_min(0) >= 1
        vertical = (torch.minimum(bottom[..., None], edges[1:]) -
                    torch.maximum(top[..., None], edges[:-1])).clamp_min(0) >= 1
        contained = (vertical[..., :, None] & horizontal[..., None, :]).flatten(-2)
        for view, value in zip(views, contained.unbind(1)):
            masks[view] = value
    return masks


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
    masks = _contained_patch_masks(boxes, coordinates, patch_counts)
    counts = torch.stack([mask.sum(-1) for mask in masks], -1)
    covering = torch.stack([
        (boxes[:, view, None, :2] <= coordinates[..., :2]).all(-1)
        & (boxes[:, view, None, 2:4] >= coordinates[..., 2:]).all(-1)
        & (counts[..., view] >= MIN_REGION_PATCHES) & positive
        for view in range(2)
    ], -1)
    # Transfer all small metadata at once, instead of three device synchronizations.
    metadata = torch.cat((coordinates, counts.float(), covering.float()), -1).detach().cpu().tolist()
    by_image = []
    for image in range(batch):
        distinct = {}
        for row, (a, b) in enumerate(pairs):
            values = metadata[image][row]
            counts_row = values[4:4 + views]
            candidates = [view for view in range(2) if values[4 + views + view]]
            if not candidates or min(counts_row[a], counts_row[b]) < MIN_REGION_PATCHES:
                continue
            teacher = max(candidates, key=lambda view: (counts_row[view], -view))
            key = tuple(values[:4])
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
        owner: (list(owner_rows[owner]) if len(owner_rows[owner]) == 1 else
                [owner_rows[owner][index] for index in torch.randperm(
            len(owner_rows[owner]), generator=generator
        ).tolist()]) for owner in images
    }
    chosen = []
    while len(chosen) < count:
        for owner in images:
            if pools[owner]:
                chosen.append(pools[owner].pop())
                if len(chosen) == count:
                    break
    return chosen


def _pool_views(views, regions, masks, required, temperature=None):
    """Batched pooling per view; only metadata loops over images/regions.

    Cap each softmax tile at 64 MiB. Recompute student softmax in backward;
    teacher pooling is detached. Zero padded region rows have zero gradients.
    """
    result = {}
    by_view = defaultdict(lambda: defaultdict(set))
    for image, index, view in required:
        by_view[view][image].add(index)
    for view, by_image in by_view.items():
        logits = views[view]
        # Student pooling tiles internally and writes one full-view gradient.
        tile = (len(by_image) if temperature is not None else
                max(1, POOL_SOFTMAX_BYTES // (logits.shape[1] * logits.shape[2] * 4)))
        images = sorted(by_image)
        for start in range(0, len(images), tile):
            selected_images = images[start:start + tile]
            indices = [sorted(by_image[image]) for image in selected_images]
            maximum = max(map(len, indices))
            rows = [[regions[image][index].row for index in selected] + [0] * (maximum - len(selected))
                    for image, selected in zip(selected_images, indices)]
            image_indices = torch.tensor(selected_images, device=logits.device)
            row_indices = torch.tensor(rows, device=logits.device)
            lengths = torch.tensor([len(selected) for selected in indices], device=logits.device)
            present = torch.arange(maximum, device=logits.device)[None] < lengths[:, None]
            weights = masks[view][image_indices[:, None], row_indices].float() * present[..., None]
            if temperature is None:
                with torch.no_grad():
                    pooled = weights @ logits.detach().index_select(0, image_indices).float()
                    pooled = pooled / weights.sum(-1, keepdim=True).clamp_min(1)
            else:
                pooled = region_probability_mean(logits, image_indices, weights, temperature, POOL_SOFTMAX_BYTES)
            # Row views do not copy their full prototype vectors.
            # Unbind once per axis: independent scalar indexing would create a
            # full padded gradient buffer for every selected row in backward.
            for image, selected, image_values in zip(selected_images, indices, pooled.unbind(0)):
                for index, value in zip(selected, image_values.unbind(0)):
                    result[image, index, view] = value
    return result


def _compare_queries(students, teachers, bank, indices):
    # NeCo-style shared GEMM: gather scalar cosines, never Q x M x D vectors.
    student_cosines = (F.normalize(students, dim=-1) @ bank.T).gather(1, indices)
    with torch.no_grad():
        teacher_cosines = (F.normalize(teachers.detach(), dim=-1) @ bank.T).gather(1, indices)
    return student_cosines, teacher_cosines


def _within_cosines(vectors, bank, packed_rows, bank_rows, output_rows, reference_columns):
    queries = F.normalize(vectors.index_select(0, packed_rows.flatten()), dim=-1).reshape(
        *packed_rows.shape, -1
    )
    references = bank[bank_rows]
    similarities = torch.bmm(queries, references.transpose(1, 2))
    return similarities.flatten(0, 1).index_select(0, output_rows).gather(1, reference_columns)


def _compare_within_queries(students, teachers, bank, queries, local_offsets, regions):
    """One bank per image, shared by all of that image's queries in the tile."""
    by_image = defaultdict(list)
    for row, query in enumerate(queries):
        by_image[query[0]].append(row)
    width = max(map(len, by_image.values()))
    packed_rows, output_positions, bank_rows = [], [0] * len(queries), []
    for group, (image, positions) in enumerate(by_image.items()):
        packed_rows.append(positions + [0] * (width - len(positions)))
        for column, row in enumerate(positions):
            output_positions[row] = group * width + column
        bank_rows.append(list(range(local_offsets[image], local_offsets[image] + len(regions[image]))))
    device = students.device
    arguments = (
        torch.tensor(packed_rows, device=device), torch.tensor(bank_rows, device=device),
        torch.tensor(output_positions, device=device),
        torch.tensor([[row - local_offsets[query[0]] for row in query[-1]] for query in queries], device=device),
    )
    student_cosines = checkpoint(_within_cosines, students, bank, *arguments, use_reentrant=False)
    with torch.no_grad():
        teacher_cosines = _within_cosines(teachers.detach(), bank, *arguments)
    return student_cosines, teacher_cosines


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

        teacher_required = set()
        for image, image_regions in enumerate(regions):
            for index, region in enumerate(image_regions):
                teacher_required.add((image, index, region.teacher_view))
                teacher_required.update((image, index, view) for view, _ in region.queries)
        teacher_pooled = _pool_views(teacher_targets, regions, masks, teacher_required)
        reference_rows, owners = [], []
        with torch.no_grad():
            for image, image_regions in enumerate(regions):
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

        zero = sum(view.reshape(-1)[:1].sum(dtype=torch.float32) * 0.0 for view in student_views)
        family_sums = [zero for _ in FAMILY_NAMES]
        family_counts = [0 for _ in FAMILY_NAMES]
        # Pool each required physical region once, batched across images.
        required = set()
        for queries in groups.values():
            for image, index, _, view, _, _ in queries:
                required.add((image, index, view))
        student_pooled = _pool_views(student_views, regions, masks, required, self.student_temperature)

        for count, queries in groups.items():
            tile = min(SORT_QUERY_CHUNK, max(1, SORT_PERMUTATION_ELEMENTS // (count * count)))
            for start in range(0, len(queries), tile):
                chunk = queries[start:start + tile]
                indices = torch.tensor([query[-1] for query in chunk], device=bank.device, dtype=torch.long)
                students = torch.stack([student_pooled[q[0], q[1], q[3]] for q in chunk])
                teachers = torch.stack([teacher_pooled[q[0], q[1], q[2]] for q in chunk])
                if self.modality == "cross_image":
                    student_cosines, teacher_cosines = _compare_queries(students, teachers, bank, indices)
                else:
                    student_cosines, teacher_cosines = _compare_within_queries(
                        students, teachers, bank, chunk, local_offsets, regions
                    )
                # Multiple directions can share one prediction or target.
                # Sort each unique (image, region, view) only once per tile.
                unique_rows, query_maps = [], []
                for view_position in (3, 2):
                    lookup, rows, mapping = {}, [], []
                    for row, query in enumerate(chunk):
                        key = (query[0], query[1], query[view_position])
                        if key not in lookup:
                            lookup[key] = len(rows)
                            rows.append(row)
                        mapping.append(lookup[key])
                    unique_rows.append(torch.tensor(rows, device=bank.device))
                    query_maps.append(torch.tensor(mapping, device=bank.device))
                with torch.no_grad():
                    _, target = bitonic_permutation(teacher_cosines.index_select(0, unique_rows[1]))
                    target = target.index_select(0, query_maps[1])
                losses = checkpoint(
                    permutation_target_cross_entropy, student_cosines.index_select(0, unique_rows[0]),
                    target, query_maps[0], use_reentrant=False,
                )
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
