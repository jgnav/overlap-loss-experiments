"""Explicit feature alternatives for controlled Probe3D comparisons."""

import hashlib

import numpy as np
import torch
from torch import nn

from evaluation.utils.common import _checkpoint_argument, _checkpoint_state, _torch_load, write_json


FEATURE_VARIANTS = ("raw_final", "final_norm_standardized", "projection", "projection_softmax", "concat_4_6_8_12")


def load_patch_projection(checkpoint_path, checkpoint_key):
    """Restore the saved patch head, including legacy iBOT shared-MLP heads."""
    from model.head import DINOHead

    checkpoint = _torch_load(checkpoint_path)
    head_state = {}
    for name, value in _checkpoint_state(checkpoint, checkpoint_key).items():
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name.startswith("head."):
            head_state[name[5:]] = value
    trunk = "patch_mlp." if any(k.startswith("patch_mlp.") for k in head_state) else "mlp."
    prototype = "last_layer2." if "last_layer2.weight_v" in head_state else "last_layer."
    if prototype + "weight_v" not in head_state:
        raise ValueError("Projection evaluation requires saved weight-normalized patch prototypes")
    weights = sorted((k for k in head_state if k.startswith(trunk) and k.endswith("weight")
                      and head_state[k].ndim == 2), key=lambda k: int(k.split(".")[1]))
    if not weights:
        raise ValueError("Projection evaluation requires the saved patch MLP")
    first, last = head_state[weights[0]], head_state[prototype + "weight_v"]
    norm = _checkpoint_argument(checkpoint, "norm_in_head")
    act = _checkpoint_argument(checkpoint, "act_in_head") or "gelu"
    last_norm = "last_norm2." if any(k.startswith("last_norm2.") for k in head_state) else "last_norm."
    if any(k.startswith(last_norm) for k in head_state):
        raise ValueError("A checkpoint with output head normalization needs a dedicated adapter")
    head = DINOHead(in_dim=first.shape[1], out_dim=last.shape[0], nlayers=len(weights),
                    hidden_dim=first.shape[0], bottleneck_dim=last.shape[1], norm=norm, act=act)
    state = {}
    for name, tensor in head_state.items():
        if name.startswith(trunk):
            state["mlp." + name[len(trunk):]] = tensor
        elif name.startswith(prototype):
            state["last_layer." + name[len(prototype):]] = tensor
    head.load_state_dict(state, strict=True)
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError("Non-finite checkpoint patch-head weights")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    head.requires_grad_(False).eval()
    return head, {"trunk": trunk, "prototype_layer": prototype, "output_dim": int(last.shape[0]),
                  "weights_sha256": digest.hexdigest(), "strict_checkpoint_load": True}


class CorrespondenceFeatures(nn.Module):
    def __init__(self, backbone, variant, patch_size, num_register_tokens=0, projection=None, temperature=1.0):
        super().__init__()
        if variant not in FEATURE_VARIANTS:
            raise ValueError(f"Unknown correspondence feature variant: {variant}")
        if temperature <= 0 or not np.isfinite(temperature):
            raise ValueError("Softmax temperature must be positive and finite")
        if variant.startswith("projection") and projection is None:
            raise ValueError("Projection feature variants require a trained patch head")
        if variant == "concat_4_6_8_12" and len(backbone.blocks) < 12:
            raise ValueError("Block concatenation requires at least twelve transformer blocks")
        self.backbone, self.variant = backbone, variant
        self.patch_size, self.num_register_tokens = patch_size, num_register_tokens
        self.projection, self.temperature = projection, temperature
        self.register_buffer("scaler_mean", None)
        self.register_buffer("scaler_scale", None)

    @torch.no_grad()
    def extract_tokens(self, images, apply_scaler=True):
        tokens = self.backbone.prepare_tokens(images)
        selected = []
        for depth, block in enumerate(self.backbone.blocks, start=1):
            tokens = block(tokens)
            if self.variant == "concat_4_6_8_12" and depth in (4, 6, 8, 12):
                selected.append(tokens[:, 1 + self.num_register_tokens:])
        if self.variant == "concat_4_6_8_12":
            # iBOT segmentation concatenates raw block outputs, without final LN.
            return torch.cat(selected, dim=-1).float()
        if self.variant == "raw_final":
            return tokens[:, 1 + self.num_register_tokens:].float()
        tokens = self.backbone.norm(tokens)[:, 1 + self.num_register_tokens:].float()
        if self.variant == "final_norm_standardized":
            if not apply_scaler:
                return tokens
            if self.scaler_mean is None:
                raise RuntimeError("StandardScaler must be fitted before correspondence evaluation")
            return (tokens - self.scaler_mean) / self.scaler_scale
        shape = tokens.shape[:2]
        # Bound hidden-layer activation memory for high-resolution patch grids.
        logits = torch.cat([self.projection(chunk) for chunk in tokens.reshape(-1, tokens.shape[-1]).split(512)])
        if self.variant == "projection_softmax":
            logits = (logits / self.temperature).softmax(dim=-1)
        return logits.reshape(*shape, -1).float()

    def extract_patch_map(self, images):
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("Correspondence image dimensions must be patch aligned")
        tokens = self.extract_tokens(images)
        height, width = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        if tokens.shape[1] != height * width:
            raise ValueError("Feature token count differs from the image patch grid")
        return tokens.transpose(1, 2).reshape(len(images), -1, height, width)


@torch.no_grad()
def fit_voc_standardizer(extractor, datasets_root, output_dir, device, seed=0, num_workers=4):
    """Fit only CAPI's 90% VOC train subset, streaming all of its patch vectors."""
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader
    from evaluation.utils.datasets import segmentation_manifest
    from evaluation.utils.dense import _build_dense_datasets, dense_resolution
    from evaluation.utils.common import print_progress

    resolution = dense_resolution(extractor.patch_size)
    train = _build_dense_datasets("pascal_voc", datasets_root, seed, resolution)["train"]
    loader = DataLoader(train, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)
    scaler = StandardScaler()
    for index, (images, _) in enumerate(loader, start=1):
        tokens = extractor.extract_tokens(images.to(device), apply_scaler=False)
        scaler.partial_fit(tokens.reshape(-1, tokens.shape[-1]).cpu().numpy())
        print_progress("StandardScaler VOC training features", index, len(loader))
    extractor.scaler_mean = torch.as_tensor(scaler.mean_, device=device, dtype=torch.float32)
    extractor.scaler_scale = torch.as_tensor(scaler.scale_, device=device, dtype=torch.float32)
    stats_path = output_dir / "standard_scaler.npz"
    np.savez(stats_path, mean=scaler.mean_, scale=scaler.scale_, variance=scaler.var_,
             samples=scaler.n_samples_seen_, image_indices=np.asarray(train.indices))
    metadata = {"type": "StandardScaler", "fit_dataset": "VOC2012 segmentation train, 90% CAPI subset",
                "fit_images": len(train), "fit_patch_vectors": int(scaler.n_samples_seen_),
                "resolution": resolution, "seed": seed, "fit_manifest": segmentation_manifest(train.dataset),
                "stats_path": str(stats_path), "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                "correspondence_test_data_used_for_fit": False,
                "application": "channelwise after final LayerNorm, before spatial interpolation/L2"}
    write_json(output_dir / "standard_scaler_protocol.json", metadata)
    return metadata
