"""CRISP adapter for post-LayerNorm final/averaged patch features.

Copied into a frozen CRISP source tree by crisp_native_features.py. The imports
below are deliberately relative to that tree's evals.models package.
"""

import torch

from .region_vits import RegionViTS
from .utils import center_padding, tokens_to_output


class NormalizedRegionViTS(RegionViTS):
    def __init__(self, checkpoint_path, feature_blocks=1, output="dense",
                 layer=-1, return_multilayer=False):
        if feature_blocks not in (1, 4) or return_multilayer or layer != -1:
            raise ValueError("Select the normalized final block or last-four average")
        super().__init__(checkpoint_path, output=output, layer=layer,
                         return_multilayer=False)
        self.feature_blocks = feature_blocks
        self.multilayers = list(range(12 - feature_blocks, 12))
        self.layer = "norm12" if feature_blocks == 1 else "norm9-12-average"
        self.checkpoint_name += "_" + self.layer

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        grid = tuple(side // self.patch_size for side in images.shape[-2:])
        layers = self.vit.get_intermediate_layers(images, n=self.feature_blocks)
        # Same patch extraction as VOC and video: the learned backbone norm is
        # applied independently to each block. Average before matching's L2 norm.
        spatial = torch.stack([layer[:, 1:].float() for layer in layers]).mean(0)
        cls_token = torch.stack([layer[:, 0].float() for layer in layers]).mean(0)
        return tokens_to_output(self.output, spatial, cls_token, grid)
