"""CPU alignment of three PCA score columns for comparable model colors."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PCAAlignment:
    projected: np.ndarray
    correlations: np.ndarray
    permutation: np.ndarray
    signs: np.ndarray


def align_pca_components(
    reference: np.ndarray, target: np.ndarray
) -> PCAAlignment:
    """Match target PCA scores to reference RGB components by Pearson correlation.

    Inputs must be [patches, 3], with corresponding image/patch positions in
    the same row order. Correlate scores, not PCA loading vectors: models may
    have different feature dimensions. Rows of the correlation matrix refer
    to reference components; columns refer to the original target components.

    Solve the optimal assignment on -abs(correlation), reorder the target,
    and flip negatively correlated matches. No rotation or rescaling is
    applied. A constant column has undefined Pearson correlation; assign it
    zero correlation so it cannot provide evidence for a match or sign flip.
    """
    reference = np.asarray(reference)
    target = np.asarray(target)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or target.shape != reference.shape
        or reference.shape[0] < 2
    ):
        raise ValueError(
            "PCA scores must have matching [patches, 3] shapes with at least "
            "two corresponding patches"
        )
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("PCA scores must contain only finite values")

    normalized = []
    for scores in (reference, target):
        centered = scores.astype(np.float64) - scores.mean(axis=0, dtype=np.float64)
        norms = np.linalg.norm(centered, axis=0)
        normalized.append(
            np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
        )
    correlations = np.clip(normalized[0].T @ normalized[1], -1.0, 1.0)

    reference_indices, target_indices = linear_sum_assignment(-np.abs(correlations))
    permutation = np.empty(3, dtype=np.intp)
    permutation[reference_indices] = target_indices
    matched_correlations = correlations[np.arange(3), permutation]
    signs = np.where(matched_correlations < 0, -1, 1).astype(np.int8)
    aligned = target[:, permutation] * signs
    return PCAAlignment(aligned, correlations, permutation, signs)
