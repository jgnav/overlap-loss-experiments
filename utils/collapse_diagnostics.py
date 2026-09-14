"""Detached epoch diagnostics; see docs/collapse_diagnostics.md for conventions."""

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.weight_norm import WeightNorm


PROTOTYPE_EPSILONS = (0.025, 0.1, 0.25, 0.5)


@torch.no_grad()
def effective_prototype_weight(layer):
    """Read effective weights without refreshing/mutating a module's cached weight.

    Legacy weight_norm caches .weight at forward time, so that attribute is
    stale after optimizer/EMA updates. compute_weight reads CURRENT g and v.
    Parametrized weight_norm computes .weight on access and needs no special case.
    """
    for hook in layer._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm) and hook.name == "weight":
            return hook.compute_weight(layer).detach()
    return layer.weight.detach()


@torch.no_grad()
def unique_prototype_counts(weights, epsilons=PROTOTYPE_EPSILONS, chunk_size=256):
    """Index-ordered representative cover with strict cosine distance < epsilon.

    Each first-uncovered prototype represents all still-uncovered neighbors
    within epsilon. This is a deterministic cover satisfying Definition 2.1,
    NOT a minimum-cardinality cover or an exact reproduction of released code.
    Each threshold has its own cover. At most chunk_size x K CPU distances
    exist at once; no K x K similarity matrix is retained on CPU or GPU.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("prototype chunk_size must be a positive integer")
    vectors = weights.detach().to(device="cpu", dtype=torch.float64)
    if vectors.ndim != 2 or min(vectors.shape) == 0:
        raise ValueError("Prototype weights must be a nonempty K x D matrix")
    norms = vectors.norm(dim=1, keepdim=True)
    if not torch.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError("Prototype geometry requires finite, nonzero vectors")
    if any(not 0 < epsilon <= 2 for epsilon in epsilons):
        raise ValueError("Prototype epsilon must be in (0, 2]")
    vectors = vectors / norms
    count = len(vectors)
    covered = np.zeros((len(epsilons), count), dtype=bool)
    counts = np.zeros(len(epsilons), dtype=np.int64)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        if covered[:, start:stop].all():
            continue
        distances = (1 - vectors[start:stop] @ vectors.T).numpy()
        for offset, index in enumerate(range(start, stop)):
            for threshold, epsilon in enumerate(epsilons):
                if not covered[threshold, index]:
                    counts[threshold] += 1
                    covered[threshold] |= distances[offset] < epsilon
                    covered[threshold, index] = True
    return {epsilon: int(value) for epsilon, value in zip(epsilons, counts)}


@torch.no_grad()
def prototype_geometry_metrics(student, teacher, chunk_size=256):
    """Measure synchronized student/EMA heads on rank zero; share scalar results."""
    distributed = dist.is_available() and dist.is_initialized()
    metrics = {}
    if not distributed or dist.get_rank() == 0:
        for name, model in (("student", student), ("teacher", teacher)):
            while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                model = model.module
            for objective, layer in model.head.prototype_layers().items():
                weights = effective_prototype_weight(layer)
                counts = unique_prototype_counts(weights, chunk_size=chunk_size)
                for epsilon, count in counts.items():
                    suffix = f"{epsilon:.3f}".replace(".", "")
                    prefix = f"{name}_{objective}"
                    metrics[f"{prefix}_unique_prototypes_eps_{suffix}"] = count
                    metrics[f"{prefix}_unique_prototype_ratio_eps_{suffix}"] = count / len(weights)
    if distributed:
        payload = [metrics]
        dist.broadcast_object_list(payload, src=0)
        metrics = payload[0]
    return metrics


class FeatureCollapseDiagnostics:
    """CPU FP64 sufficient statistics of detached, post-backbone patch features."""

    def __init__(self, dimension, max_features_per_batch=4096):
        if type(max_features_per_batch) is not int or max_features_per_batch < 2:
            raise ValueError("max_features_per_batch must be an integer >= 2")
        self.dimension = dimension
        self.max_features_per_batch = max_features_per_batch
        self.count = torch.zeros((), dtype=torch.float64)
        self.sum_features = torch.zeros(dimension, dtype=torch.float64)
        self.sum_outer = torch.zeros(dimension, dimension, dtype=torch.float64)
        self.sum_directions = torch.zeros(dimension, dtype=torch.float64)

    @torch.no_grad()
    def update(self, patch_features):
        """Evenly sample flattened view/image/patch positions without RNG or copies
        of the full feature grid. The caller excludes CLS before entering here.
        """
        if patch_features.ndim != 3 or patch_features.shape[-1] != self.dimension:
            raise ValueError("Expected [views * batch, patches, embedding_dimension]")
        total = patch_features.shape[0] * patch_features.shape[1]
        take = min(total, self.max_features_per_batch)
        if take == 0:
            return
        indices = torch.arange(take, device=patch_features.device) * total // take
        patches = patch_features.shape[1]
        features = patch_features.detach()[indices // patches, indices % patches]
        features = features.to(device="cpu", dtype=torch.float64)
        norms = features.norm(dim=-1, keepdim=True)
        directions = features / norms.clamp_min(torch.finfo(torch.float64).tiny)
        self.count += take
        self.sum_features += features.sum(0)
        self.sum_outer += features.T @ features
        self.sum_directions += directions.sum(0)

    @torch.no_grad()
    def compute(self, device=None):
        """All-reduce statistics FIRST, then form covariance and its spectrum."""
        dimension = self.dimension
        statistics = torch.cat((
            self.count.reshape(1), self.sum_features,
            self.sum_outer.flatten(), self.sum_directions,
        ))
        if dist.is_available() and dist.is_initialized():
            if dist.get_backend() == "nccl":
                statistics = statistics.to(device)
            dist.all_reduce(statistics)
            statistics = statistics.cpu()
        count = statistics[0].item()
        prefix = "teacher_patch_feature_"
        result = {
            prefix + "effective_rank": 0.0,
            prefix + "effective_rank_ratio": 0.0,
            prefix + "top1_variance_fraction": 0.0,
            prefix + "mean_direction_norm": 0.0,
            prefix + "sample_count": int(count),
        }
        if count == 0:
            return result
        mean = statistics[1:1 + dimension] / count
        second_moment = statistics[1 + dimension:1 + dimension + dimension**2]
        second_moment = second_moment.reshape(dimension, dimension) / count
        covariance = second_moment - torch.outer(mean, mean)
        eigenvalues = torch.linalg.eigvalsh((covariance + covariance.T) / 2).clamp_min(0)
        variance = eigenvalues.sum()
        # A constant feature cloud may leave a tiny positive subtraction residual.
        tolerance = 10 * dimension * torch.finfo(torch.float64).eps * second_moment.abs().max()
        if variance > tolerance:
            probabilities = eigenvalues / variance
            positive = probabilities[probabilities > 0]
            effective_rank = (-(positive * positive.log()).sum()).exp().item()
            result[prefix + "effective_rank"] = effective_rank
            result[prefix + "effective_rank_ratio"] = effective_rank / dimension
            result[prefix + "top1_variance_fraction"] = (eigenvalues[-1] / variance).item()
        direction = statistics[-dimension:] / count
        result[prefix + "mean_direction_norm"] = direction.norm().clamp(0, 1).item()
        return result
