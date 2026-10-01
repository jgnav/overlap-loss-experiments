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
