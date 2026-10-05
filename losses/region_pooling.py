"""Tiled softmax/mean pooling with one gradient allocation per crop view."""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _RegionProbabilityMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, image_indices, weights, temperature, tile):
        # Geometry supplies unique image indices and detached hard patch masks.
        ctx.save_for_backward(logits, image_indices, weights)
        ctx.temperature, ctx.tile = temperature, tile
        result = logits.new_empty((len(image_indices), weights.shape[1], logits.shape[2]), dtype=torch.float32)
        for start in range(0, len(image_indices), tile):
            images = image_indices[start:start + tile]
            selected = weights[start:start + tile]
            probabilities = F.softmax(logits.index_select(0, images).float() / temperature, -1)
            result[start:start + tile] = torch.bmm(selected, probabilities) / selected.sum(-1, keepdim=True).clamp_min(1)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, region_gradient):
        logits, image_indices, weights = ctx.saved_tensors
        # Every selected image occurs once. Avoid the batch-sized temporary
        # gradient created by an individual Select/IndexSelectBackward per tile.
        gradient = (torch.empty_like(logits) if len(image_indices) == len(logits)
                    else torch.zeros_like(logits))
        for start in range(0, len(image_indices), ctx.tile):
            images = image_indices[start:start + ctx.tile]
            selected = weights[start:start + ctx.tile]
            probabilities = F.softmax(logits.index_select(0, images).float() / ctx.temperature, -1)
            pooled_gradient = region_gradient[start:start + ctx.tile].float()
            patch_gradient = torch.bmm(
                selected.transpose(1, 2),
                pooled_gradient / selected.sum(-1, keepdim=True).clamp_min(1),
            )
            softmax_gradient = probabilities * (
                patch_gradient - (patch_gradient * probabilities).sum(-1, keepdim=True)
            ) / ctx.temperature
            gradient.index_copy_(0, images, softmax_gradient.to(logits.dtype))
        return gradient, None, None, None, None


def region_probability_mean(logits, image_indices, weights, temperature, byte_budget):
    """Exact arithmetic mean of selected softmax distributions, in float32.

    Image indices must be unique. Recompute softmax tiles in backward instead
    of retaining patch probabilities or constructing one autograd path/image.
    This training operation supports first-order gradients.
    """
    tile = max(1, byte_budget // (logits.shape[1] * logits.shape[2] * 4))
    return _RegionProbabilityMean.apply(logits, image_indices, weights.detach().float(), float(temperature), tile)


class _RegionLogProbabilityMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, image_indices, weights, temperature, tile):
        # Keep tiny student probabilities in log space: clamping an ordinary
        # probability mean would truncate CE and lose gradients for wrong,
        # confident predictions. Regions are processed in bounded tiles.
        result = logits.new_empty((len(image_indices), weights.shape[1], logits.shape[2]), dtype=torch.float32)
        for start in range(0, len(image_indices), tile):
            images = image_indices[start:start + tile]
            selected = weights[start:start + tile]
            logs = F.log_softmax(logits.index_select(0, images).float() / temperature, -1)
            for region in range(weights.shape[1]):
                w = selected[:, region]
                mass = w.sum(-1, keepdim=True)
                pooled = torch.logsumexp(logs + w.log()[..., None], 1) - mass.clamp_min(1).log()
                result[start:start + tile, region] = torch.where(mass > 0, pooled, 0.)
        ctx.save_for_backward(logits, image_indices, weights, result)
        ctx.temperature, ctx.tile = temperature, tile
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, region_gradient):
        logits, image_indices, weights, pooled_logs = ctx.saved_tensors
        gradient = (torch.empty_like(logits) if len(image_indices) == len(logits)
                    else torch.zeros_like(logits))
        for start in range(0, len(image_indices), ctx.tile):
            images = image_indices[start:start + ctx.tile]
            selected = weights[start:start + ctx.tile]
            logs = F.log_softmax(logits.index_select(0, images).float() / ctx.temperature, -1)
            log_gradient = torch.zeros_like(logs)
            for region in range(weights.shape[1]):
                w = selected[:, region]
                # For each prototype this posterior distributes the pooled
                # gradient among contributing patches; it cannot underflow
                # merely because the prototype's overall probability is tiny.
                posterior = (logs + w.log()[..., None]
                             - w.sum(-1, keepdim=True).clamp_min(1).log()[..., None]
                             - pooled_logs[start:start + ctx.tile, region, None]).exp()
                log_gradient.add_(posterior * region_gradient[start:start + ctx.tile, region, None].float())
            softmax_gradient = (log_gradient - logs.exp() * log_gradient.sum(-1, keepdim=True)) / ctx.temperature
            gradient.index_copy_(0, images, softmax_gradient.to(logits.dtype))
        return gradient, None, None, None, None


def region_log_probability_mean(logits, image_indices, weights, temperature, byte_budget):
    """Stable log of grouped probability means with tiled first-order backward.

    Image indices must be unique; weights have [images, regions, patches]
    shape. Empty regions return connected zeros and contribute zero gradients.
    """
    tile = max(1, byte_budget // (logits.shape[1] * logits.shape[2] * 4))
    return _RegionLogProbabilityMean.apply(logits, image_indices, weights.detach().float(), float(temperature), tile)
