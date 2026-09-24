"""DINOv2-style Kozachenko-Leonenko regularizer for student CLS features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """Negative log distance to the nearest other sample on the unit sphere."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.distance = nn.PairwiseDistance(p=2, eps=eps)

    def forward(self, features):
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("KoLeo requires at least two [batch, features] rows")
        # The outer training autocast may be FP16/BF16; nearest-neighbor
        # selection and log distances must use float32.
        with torch.autocast(device_type=features.device.type, enabled=False):
            vectors = F.normalize(features.float(), p=2, dim=-1, eps=self.eps)
            with torch.no_grad():
                similarities = vectors @ vectors.T
                similarities.fill_diagonal_(-torch.inf)
                neighbors = similarities.argmax(dim=1)
            distances = self.distance(vectors, vectors[neighbors])
            return -(distances + self.eps).log().mean()
