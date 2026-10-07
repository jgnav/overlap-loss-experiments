"""CRISP-inspired, one-block masked aggregation of backbone patch features."""

import torch
from torch import nn

from .vision_transformer import Block
from utils.training import trunc_normal_


class RegionTokenAggregation(nn.Module):
    """A learned query reads only selected patches; patch queries are unused.

    This computes the concept-token row of CRISP's masked transformer without
    constructing all-masked patch-query rows. CLS/register/self keys are absent.
    Backbone patches already carry positional information. Area weights, when
    configured, act as attention priors; binary selection adds no prior.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, dim))
        self.block = Block(dim, num_heads, mlp_ratio=4., qkv_bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)
        trunc_normal_(self.token, std=.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, patches, weights):
        if patches.ndim != 3 or weights.shape != patches.shape[:2]:
            raise ValueError("Region token weights must match [batch, patches] features")
        selected = weights > 0
        valid = selected.any(-1)
        # Excluded values cannot leak through attention or receive gradients.
        patches = patches.masked_fill(~selected[..., None], 0)
        token = self.token.expand(len(patches), -1, -1)
        sequence = self.block.norm1(torch.cat((token, patches), dim=1))
        batch, length, dim = sequence.shape
        attention = self.block.attn
        qkv = attention.qkv(sequence).reshape(
            batch, length, 3, attention.num_heads, dim // attention.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, keys, values = qkv[0, :, :, :1], qkv[1, :, :, 1:], qkv[2, :, :, 1:]
        scores = (query @ keys.transpose(-2, -1)).float() * attention.scale
        # Give empty rows a harmless zero-feature key, then zero their output.
        # Every parameter remains in the graph on empty DDP ranks.
        safe_weights = weights.float().clone()
        safe_weights[~valid, 0] = 1.
        scores = scores + safe_weights.log()[:, None, None, :]
        probabilities = attention.attn_drop(scores.softmax(-1).to(values.dtype))
        pooled = (probabilities @ values).transpose(1, 2).reshape(batch, 1, dim)
        token = token + attention.proj_drop(attention.proj(pooled))
        token = token + self.block.mlp(self.block.norm2(token))
        return self.norm(token[:, 0]).masked_fill(~valid[:, None], 0)
