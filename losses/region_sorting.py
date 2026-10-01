"""NeCo's bitonic sorter, using indexed comparators instead of dense wiring.

Matches diffsort 0.2.0 (the version pinned by NeCo): Cauchy interpolation,
steepness 100, and removal of the leading unused wires for arbitrary sizes.
Permutation axes are [batch, reference identity, rank], as in diffsort.
See third_party/diffsort_LICENSE for the upstream MIT license.
"""

from functools import lru_cache
import math

import torch


@lru_cache(maxsize=128)
def _bitonic_stages(size, device):
    blocks = (size - 1).bit_length()
    offset = 2 ** blocks - size
    stages = []
    for block in range(blocks):
        for layer in range(block + 1):
            stride = 2 ** (block - layer)
            first, second, low, high = [], [], [], []
            for start in range(0, 2 ** blocks, 2 * stride):
                for index in range(start, start + stride):
                    a, b = index, index + stride
                    if a < offset or b < offset:
                        # diffsort leaves the one surviving wire unchanged.
                        continue
                    a, b = a - offset, b - offset
                    ascending = (index // 2 ** (block + 1)) % 2 == 0
                    first.append(a)
                    second.append(b)
                    low.append(a if ascending else b)
                    high.append(b if ascending else a)
            stages.append(tuple(torch.tensor(x, device=device, dtype=torch.long)
                                for x in (first, second, low, high)))
    return stages


@lru_cache(maxsize=128)
def _bitonic_wires(size, device):
    """Each stage is a full wire permutation, so no scatter/copies are needed."""
    wires = []
    for first, second, low, high in _bitonic_stages(size, device):
        partner = torch.arange(size, device=device)
        partner[first], partner[second] = second, first
        orientation = torch.zeros(size, device=device)
        orientation[low], orientation[high] = 1., -1.
        wires.append((partner, orientation))
    return wires


def bitonic_permutation(vectors):
    """Return sorted values and the complete soft permutation (no padding ranks)."""
    if vectors.ndim != 2 or vectors.shape[1] < 2:
        raise ValueError("Bitonic sorting requires [queries, references] with at least two references")
    values = vectors.float()
    size = values.shape[1]
    permutation = torch.eye(size, device=values.device, dtype=values.dtype).expand(
        len(values), -1, -1
    )
    for partner, orientation in _bitonic_wires(size, str(values.device)):
        other = values.index_select(1, partner)
        keep = torch.atan(100.0 * (other - values) * orientation) / math.pi + .5
        weight = keep[:, None]
        permutation = weight * permutation + (1 - weight) * permutation.index_select(2, partner)
        values = keep * values + (1 - keep) * other
    return values, permutation


def permutation_cross_entropy(student_similarities, teacher_similarities):
    """Teacher-to-student CE: sum identities, average ranks, one value/query."""
    with torch.no_grad():
        _, target = bitonic_permutation(teacher_similarities.detach())
    return permutation_target_cross_entropy(student_similarities, target)


def permutation_target_cross_entropy(student_similarities, target, query_indices=None):
    """A precomputed detached teacher target avoids teacher backward recomputation."""
    _, prediction = bitonic_permutation(student_similarities)
    if query_indices is not None:
        prediction = prediction.index_select(0, query_indices)
    return -(target.detach() * prediction.clamp_min(1e-12).log()).sum(1).mean(1)
