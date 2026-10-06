#!/usr/bin/env python3
"""Individual PCA panels for iBOT, DINOv2, DINOv3, and regional iBOT.

Hard-coded configuration; launch in the repository's training environment:
  sbatch --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
    --wrap="srun ./.conda-env/bin/python -u pca_visualization.py"

The six Figure 13 inputs are supplied next to this script. Outputs are flat,
lossless PNG panels and PCA scores/metadata for later assembly in LaTeX.
No combined figure is created. Official DINO code is fetched once into the
torch.hub cache, or supplied as local checkouts below. DINOv2 weights can be
downloaded automatically. Obtain DINOv3 weights from its official release
and place them at DINOV3_CHECKPOINT before submitting an offline GPU job.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from utils.pca_alignment import align_pca_components


# ---- Hard-coded configuration -----------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
IMAGENET_VAL = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/datasets/imagenet/val")
OUTPUT_DIR = REPO_ROOT / "output/pca_visualizations_1000"
INPUT_IMAGE = None                       # optional --image bypasses dataset sampling
IMAGENET_MATCH = None                    # optional exact joint-view PCA protocol
COLOR_REFERENCE = None                   # optional existing iBOT PCA panel
COLOR_MAPPING_SOURCE = None              # optional recovered palette from another image
PCA_COLOR_PROTOCOL = "sigmoid_whitened"
LINEAR_COLOR_SCALE = None
DINOV3_CHECKPOINT = REPO_ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
CHECKPOINTS = {
    "iBOT": REPO_ROOT / "checkpoints/ibot_vit_small.pth",
    # If absent, download the official released weights through torch.hub.
    "DINOv2": REPO_ROOT / "checkpoints/dinov2_vits14_reg4_pretrain.pth",
    "DINOv3": DINOV3_CHECKPOINT,
    "Ours": REPO_ROOT / "output/long_ibot_vit_small/85535_0/checkpoint_source1000_continuation0200.pth",
}
CHECKPOINT_KEY = "teacher"                # iBOT and Ours only
PCA_REFERENCE_MODEL = "iBOT"
MODEL_FAMILIES = {"iBOT": "ibot", "DINOv2": "dinov2", "DINOv3": "dinov3", "Ours": "ibot"}
PATCH_SIZES = {"iBOT": 16, "DINOv2": 14, "DINOv3": 16, "Ours": 16}
HUB_MODELS = {"DINOv2": "dinov2_vits14_reg", "DINOv3": "dinov3_vits16"}
# Pin implementation revisions to make later runs reproducible.
HUB_REPOS = {
    "DINOv2": "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    "DINOv3": "facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
}
# Optional official repository checkouts for jobs without internet access.
LOCAL_DINO_REPOS = {"DINOv2": None, "DINOv3": None}
ALLOW_DINOV2_WEIGHT_DOWNLOAD = True
N_IMAGES = 1000
SEED = 2
# Exclude the previous cosine script's 100 samples, even if it has not run yet.
PREVIOUS_COSINE_SEED = 1
PREVIOUS_COSINE_N_IMAGES = 100
PREVIOUS_COSINE_MANIFEST = REPO_ROOT / "output/patch_cosine_similarity/manifest.json"
VIS_RESOLUTION = 896                     # divisible by both 14 and 16
# Retain the previous script's four-layer averaging for ALL models.
# Set to 1 for a final-block-only comparison like the DINOv3 paper.
N_LAST_LAYERS = 4
SIGMOID_GAIN = 1.5                       # shared whitened-PCA contrast
DEVICE = torch.device("cuda")
DPI = 300
OVERWRITE = False
# Show each model's actual patch grid, without smoothing.
MAP_RESAMPLING = Image.Resampling.NEAREST
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
SOURCE_BASE = "https://arxiv.org/html/2508.10104v1/images/pca_comparison"
EXAMPLES = (
    ("dinosaur", 3), ("bicycle", 2), ("garden", 1),
    ("monkey", 4), ("flowers", 0), ("antelopes", 5),
)
# -----------------------------------------------------------------------------


def MODEL_TRANSFORM(image: Image.Image) -> torch.Tensor:
    """Identical RGB/ImageNet normalization for every model, without torchvision."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _check_inputs() -> None:
    if tuple(CHECKPOINTS) != ("iBOT", "DINOv2", "DINOv3", "Ours"):
        raise ValueError("CHECKPOINTS must contain iBOT, DINOv2, DINOv3, Ours in that order")
    if PCA_REFERENCE_MODEL not in CHECKPOINTS:
        raise ValueError("PCA_REFERENCE_MODEL must name an entry in CHECKPOINTS")
    if type(VIS_RESOLUTION) is not int or VIS_RESOLUTION < 224 or VIS_RESOLUTION % 112:
        raise ValueError("VIS_RESOLUTION must be >=224 and divisible by 112 (14 and 16)")
    if type(N_IMAGES) is not int or N_IMAGES < 1:
        raise ValueError("N_IMAGES must be a positive integer")
    if type(N_LAST_LAYERS) is not int or not 1 <= N_LAST_LAYERS <= 12:
        raise ValueError("N_LAST_LAYERS must be an integer between 1 and 12")
    if not math.isfinite(SIGMOID_GAIN) or SIGMOID_GAIN <= 0:
        raise ValueError("SIGMOID_GAIN must be positive and finite")
    if CHECKPOINT_KEY not in ("teacher", "student"):
        raise ValueError("CHECKPOINT_KEY must be teacher or student")
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the configured DEVICE")
    missing = [str(CHECKPOINTS[name]) for name in ("iBOT", "DINOv3", "Ours") if not CHECKPOINTS[name].is_file()]
    if not ALLOW_DINOV2_WEIGHT_DOWNLOAD and not CHECKPOINTS["DINOv2"].is_file():
        missing.append(str(CHECKPOINTS["DINOv2"]))
    if missing:
        raise FileNotFoundError(
            "Missing checkpoints:\n  " + "\n  ".join(missing)
            + "\nDINOv3 ViT-S/16 weights: https://github.com/facebookresearch/dinov3#pretrained-models"
        )
    for name, checkout in LOCAL_DINO_REPOS.items():
        if checkout is not None and not (Path(checkout) / "hubconf.py").is_file():
            raise FileNotFoundError(f"{name} checkout has no hubconf.py: {checkout}")
    if INPUT_IMAGE is not None:
        if not INPUT_IMAGE.is_file():
            raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")
        with Image.open(INPUT_IMAGE) as image:
            image.verify()
    else:
        for name, _ in EXAMPLES:
            if not (REPO_ROOT / f"pca_{name}.png").is_file():
                raise FileNotFoundError(f"Missing Figure 13 input: pca_{name}.png")
        if not IMAGENET_VAL.is_dir():
            raise FileNotFoundError(f"ImageNet validation ImageFolder not found: {IMAGENET_VAL}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not OVERWRITE:
        raise FileExistsError(f"Output folder is not empty: {OUTPUT_DIR}. Change OUTPUT_DIR or set OVERWRITE=True")


def _imagenet_samples() -> list[tuple[Path, int, str]]:
    """Match ImageFolder's sorted class/file enumeration without loading pixels."""
    classes = sorted(path.name for path in IMAGENET_VAL.iterdir() if path.is_dir())
    samples = []
    for class_index, class_name in enumerate(classes):
        for directory, _, files in sorted(os.walk(IMAGENET_VAL / class_name, followlinks=True)):
            for filename in sorted(files):
                path = Path(directory) / filename
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, class_index, class_name))
    if not samples:
        raise ValueError("ImageNet must contain class subdirectories with supported images")
    return samples


def _sample_images() -> list[dict]:
    """Six published examples plus N_IMAGES reproducible ImageNet images."""
    if INPUT_IMAGE is not None:
        return [{
            "name": INPUT_IMAGE.stem.removesuffix("_original"), "source": "uploaded",
            "source_path": str(INPUT_IMAGE),
            "preprocessing": "preserve whole image; resize to common patch-size multiples",
        }]
    records = [{
        "name": f"fig13_{number:02d}_{name}", "source": "dinov3_figure13",
        "source_path": str(REPO_ROOT / f"pca_{name}.png"),
        "source_url": f"{SOURCE_BASE}/img_{source_index}.lr.jpg",
        "preprocessing": "preserve 4:3 view; resize to common patch-size multiples",
    } for number, (name, source_index) in enumerate(EXAMPLES, 1)]
    samples = _imagenet_samples()
    previous_paths = set()
    if PREVIOUS_COSINE_MANIFEST.is_file():
        previous = json.loads(PREVIOUS_COSINE_MANIFEST.read_text())
        # A previous run may have used different sampling settings.
        previous_paths = {
            str(Path(record["source_path"]).resolve()) for record in previous["images"]
            if record.get("source") == "imagenet" or "dataset_index" in record
        }
    excluded = {index for index, (path, _, _) in enumerate(samples) if str(path.resolve()) in previous_paths}
    # Reconstruct the cosine script's default selection if no manifest exists.
    if not previous_paths:
        count = min(PREVIOUS_COSINE_N_IMAGES, len(samples))
        excluded = set(np.random.default_rng(PREVIOUS_COSINE_SEED).choice(len(samples), size=count, replace=False).tolist())
    candidates = np.array([index for index in range(len(samples)) if index not in excluded], dtype=np.int64)
    if len(candidates) < N_IMAGES:
        raise ValueError(f"Need {N_IMAGES} new ImageNet images; only {len(candidates)} remain after excluding previous samples")
    indices = np.random.default_rng(SEED).choice(candidates, size=N_IMAGES, replace=False)
    for number, index in enumerate(indices, 1):
        path, class_index, class_name = samples[int(index)]
        records.append({
            "name": f"imagenet_{number:03d}_{class_name}_{path.stem}", "source": "imagenet",
            "source_path": str(path), "dataset_index": int(index),
            "class_index": class_index, "class_name": class_name,
            "preprocessing": "bicubic resize shorter side, then center crop",
        })
    return records


def _prepare_image(record: dict) -> Image.Image:
    with Image.open(record["source_path"]) as source:
        image = source.convert("RGB")
    width, height = image.size
    if IMAGENET_MATCH is not None:
        if image.size != (VIS_RESOLUTION, VIS_RESOLUTION):
            raise ValueError("ImageNet matching requires the exported square *_original.png at its native resolution")
        return image
    if record["source"] in ("dinov3_figure13", "uploaded"):
        # Preserve the whole example instead of cutting away the dinosaur/bicycle.
        scale = VIS_RESOLUTION / max(width, height)
        size = tuple(max(112, round(dimension * scale / 112) * 112) for dimension in (width, height))
        return image.resize(size, Image.Resampling.BICUBIC)
    if width <= height:
        size = (VIS_RESOLUTION, int(VIS_RESOLUTION * height / width))
    else:
        size = (int(VIS_RESOLUTION * width / height), VIS_RESOLUTION)
    if image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    left = round((image.width - VIS_RESOLUTION) / 2)
    top = round((image.height - VIS_RESOLUTION) / 2)
    return image.crop((left, top, left + VIS_RESOLUTION, top + VIS_RESOLUTION))


def _official_dino_constructor(name):
    """Import only the backbone factory, avoiding unrelated hub evaluation deps."""
    checkout = LOCAL_DINO_REPOS[name]
    package = MODEL_FAMILIES[name]
    if checkout is None:
        owner_repo, revision = HUB_REPOS[name].split(":")
        cache = Path(torch.hub.get_dir())
        cache.mkdir(parents=True, exist_ok=True)
        checkout = cache / f"pca_{package}_{revision}"
        if not (checkout / package / "hub/backbones.py").is_file():
            # Fetch an immutable official source archive into the cache, never the repo.
            with tempfile.TemporaryDirectory(prefix="pca-model-", dir=cache) as temporary:
                archive = Path(temporary) / "source.zip"
                torch.hub.download_url_to_file(
                    f"https://codeload.github.com/{owner_repo}/zip/{revision}", str(archive)
                )
                with zipfile.ZipFile(archive) as source:
                    for member in source.infolist():
                        destination = (Path(temporary) / member.filename).resolve()
                        if not destination.is_relative_to(Path(temporary).resolve()):
                            raise ValueError("Unexpected path in official model archive")
                    source.extractall(temporary)
                extracted = Path(temporary) / f"{package}-{revision}"
                if not (extracted / package / "hub/backbones.py").is_file():
                    raise ValueError("Official source archive has no backbone implementation")
                # Another job may have populated the same immutable cache concurrently.
                if not checkout.exists():
                    try:
                        extracted.rename(checkout)
                    except OSError:
                        if not (checkout / package / "hub/backbones.py").is_file():
                            raise
    checkout = Path(checkout).resolve()
    existing = sys.modules.get(package)
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError(f"{package} already imported from a different checkout")
    sys.path.insert(0, str(checkout))
    try:
        factories = importlib.import_module(f"{package}.hub.backbones")
    finally:
        sys.path.remove(str(checkout))
    return getattr(factories, HUB_MODELS[name])


def _load_model(name: str) -> tuple[torch.nn.Module, dict]:
    path = CHECKPOINTS[name]
    if MODEL_FAMILIES[name] == "ibot":
        from evaluation.utils.common import load_backbone
        model, metadata = load_backbone(path, CHECKPOINT_KEY, "vit_small")
        if metadata["patch_size"] != PATCH_SIZES[name]:
            raise ValueError(f"Unexpected {name} patch size: {metadata['patch_size']}")
        return model, metadata
    checkout = LOCAL_DINO_REPOS[name]
    repository = str(checkout) if checkout is not None else HUB_REPOS[name]
    # iBOT's loader cannot handle RoPE or DINO LayerScale: use official code.
    local_weights = path.is_file()
    if name == "DINOv3" and not local_weights:
        raise FileNotFoundError(f"Missing DINOv3 weights: {path}")
    if name == "DINOv2" and not local_weights and not ALLOW_DINOV2_WEIGHT_DOWNLOAD:
        raise FileNotFoundError(f"Missing DINOv2 weights: {path}")
    constructor = _official_dino_constructor(name)
    model = constructor(pretrained=(name == "DINOv2" and not local_weights))
    if local_weights:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError(f"{name} needs the official released backbone state dict: {path}")
        model.load_state_dict(state, strict=True)
    if int(model.patch_size) != PATCH_SIZES[name] or model.embed_dim != 384:
        raise ValueError(f"{name} constructor did not produce the configured ViT-S")
    model.eval().requires_grad_(False)
    metadata = {
        "architecture": HUB_MODELS[name], "patch_size": int(model.patch_size),
        "feature_dimension": int(model.embed_dim), "implementation": repository,
        "checkpoint": str(path) if local_weights else "official torch.hub pretrained weights",
        "register_tokens": int(getattr(model, "num_register_tokens", getattr(model, "n_storage_tokens", 0))),
    }
    if local_weights:
        metadata.update(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    return model, metadata


@torch.inference_mode()
def _extract_dense_features(model, image, family="ibot", patch_size=None) -> np.ndarray:
    """Average the last N normalized block outputs, with only spatial tokens."""
    depth = model.get_num_layers() if family == "ibot" else len(model.blocks)
    if type(N_LAST_LAYERS) is not int or not 1 <= N_LAST_LAYERS <= depth:
        raise ValueError(f"N_LAST_LAYERS must be an integer between 1 and {depth}")
    tensor = MODEL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    if family == "ibot":
        # This repository's API includes CLS but already excludes registers.
        layers = [layer[:, 1:] for layer in model.get_intermediate_layers(tensor, n=N_LAST_LAYERS)]
    else:
        # Official DINO APIs already exclude CLS and register/storage tokens.
        layers = model.get_intermediate_layers(tensor, n=N_LAST_LAYERS, reshape=False, return_class_token=False, norm=True)
    if len(layers) != N_LAST_LAYERS:
        raise ValueError("Backbone did not return the requested number of blocks")
    features = torch.stack([layer.float() for layer in layers]).mean(0).squeeze(0).cpu().numpy()
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected finite [spatial_patches, feature_dimension] features")
    if patch_size is None:
        if math.isqrt(features.shape[0]) ** 2 != features.shape[0]:
            raise ValueError(f"Non-square patch-token grid with {features.shape[0]} tokens")
    elif features.shape[0] != (image.height // patch_size) * (image.width // patch_size):
        raise ValueError("Unexpected patch count: CLS/register tokens or image borders were included")
    return features


def _orient_component_signs(projected: np.ndarray) -> np.ndarray:
    """Keep the existing positive-skew convention for the reference colors."""
    projected = projected.copy()
    for index in range(3):
        component = projected[:, index].astype(np.float64)
        if np.mean(component ** 3) < 0:
            projected[:, index] *= -1
    return projected


def _fit_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if min(features.shape) < 3 or not np.isfinite(features).all():
        raise ValueError("PCA needs at least three finite patch features/dimensions")
    pca = PCA(n_components=3, whiten=True, svd_solver="full")
    scores = pca.fit_transform(features).astype(np.float32)
    if not np.isfinite(scores).all() or np.any(pca.explained_variance_ <= np.finfo(np.float32).eps):
        raise ValueError("Degenerate top-three PCA components; inspect the extracted features")
    return scores, pca.explained_variance_ratio_.astype(np.float32)


def _configure_imagenet_match(protocol_path: Path | None) -> None:
    """Match original-view exports to imagenet_visualizations.py's joint PCA."""
    global IMAGENET_MATCH, VIS_RESOLUTION, N_LAST_LAYERS, SIGMOID_GAIN, SEED, MODEL_TRANSFORM
    import imagenet_visualizations as imagenet
    protocol = json.loads(protocol_path.read_text()) if protocol_path is not None else None
    if protocol is not None:
        VIS_RESOLUTION = int(protocol["resolution"])
        import re
        layers = re.search(r"mean of last (\d+) normalized", protocol["feature"])
        if layers is None:
            raise ValueError("Cannot identify feature layers in ImageNet protocol")
        N_LAST_LAYERS = int(layers.group(1))
        imagenet.VIEW_CROP_FRACTION = float(protocol["view_crop_fraction"])
        SEED = int(protocol["seed"])
        if "joint whitened PCA" not in protocol["pca"]:
            raise ValueError("ImageNet protocol does not describe the supported joint-view PCA")
        for name in ("iBOT", "Ours"):
            CHECKPOINTS[name] = Path(protocol["checkpoints"][name]["checkpoint"]).expanduser().resolve()
    else:
        VIS_RESOLUTION = imagenet.VIS_RESOLUTION
        N_LAST_LAYERS = imagenet.N_LAST_LAYERS
        SEED = imagenet.SEED
        for name in ("iBOT", "Ours"):
            CHECKPOINTS[name] = imagenet.CHECKPOINTS[name]
    SIGMOID_GAIN = imagenet.PCA_SIGMOID_GAIN
    imagenet.VIS_RESOLUTION = VIS_RESOLUTION
    imagenet.N_LAST_LAYERS = N_LAST_LAYERS
    imagenet.DEVICE = DEVICE
    MODEL_TRANSFORM = imagenet.MODEL_TRANSFORM
    IMAGENET_MATCH = {
        "implementation": imagenet,
        "protocol": protocol,
        "protocol_path": str(protocol_path) if protocol_path is not None else None,
        "settings_source": "saved ImageNet protocol" if protocol is not None else "current ImageNet script checkpoints and settings",
        "view_crop_fraction": imagenet.VIEW_CROP_FRACTION,
    }


def _fit_imagenet_pca(model, image: Image.Image, name: str):
    """Use the same three views, feature normalization, and PCA sign convention."""
    imagenet = IMAGENET_MATCH["implementation"]
    views, _ = imagenet.make_views(image)
    tokens = {}
    for view_name in imagenet.VIEW_NAMES:
        if MODEL_FAMILIES[name] == "ibot":
            tokens[view_name] = imagenet.extract_tokens(model, views[view_name], PATCH_SIZES[name])
        else:
            tokens[view_name] = SimpleNamespace(patches=_extract_dense_features(
                model, views[view_name], MODEL_FAMILIES[name], PATCH_SIZES[name]
            ))
    features = np.concatenate([tokens[view].patches for view in imagenet.VIEW_NAMES])
    if PCA_COLOR_PROTOCOL == "legacy_linear_l2":
        features = features.astype(np.float64)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("Legacy PCA needs finite nonzero patch features")
        normalized = features / norms
        centered = normalized - normalized.mean(axis=0, keepdims=True)
        pca = PCA(n_components=3, whiten=False, svd_solver="full")
        scores = pca.fit_transform(centered)
        for component in range(3):
            column = scores[:, component]
            second_moment = np.mean(column ** 2)
            if second_moment > 0 and np.mean(column ** 3) / second_moment ** 1.5 < 0:
                scores[:, component] *= -1
        return scores[:len(tokens["original"].patches)].astype(np.float32), pca.explained_variance_ratio_.astype(np.float32)
    scores = imagenet.pca_triplet(tokens)["original"]
    _, variance = _fit_pca(features)
    return scores, variance


def _project_pca(features: np.ndarray) -> np.ndarray:
    return _fit_pca(features)[0]


def _resample_scores(scores: np.ndarray, grid: tuple, target_grid: tuple) -> np.ndarray:
    """Compare patch-center score fields in shared normalized image coordinates."""
    if grid == target_grid:
        return scores
    tensor = torch.from_numpy(scores.reshape(*grid, 3).transpose(2, 0, 1).copy()).unsqueeze(0)
    # Interpolation is for color matching only, never for the exported map.
    resized = F.interpolate(tensor, size=target_grid, mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).reshape(-1, 3).numpy()


def _align_scores(reference, reference_grid, scores, grid):
    alignment = align_pca_components(reference, _resample_scores(scores, grid, reference_grid))
    aligned = scores[:, alignment.permutation] * alignment.signs
    matched = alignment.correlations[np.arange(3), alignment.permutation]
    return aligned, {
        "correlation_matrix": alignment.correlations.tolist(),
        "target_components_zero_based": alignment.permutation.tolist(),
        "signs": alignment.signs.tolist(),
        "matched_correlations_before_sign_flip": matched.tolist(),
        "mean_absolute_matched_correlation": float(np.abs(matched).mean()),
    }


def _match_reference_colors(scores, grid, reference_path: Path):
    """Recover only PCA channel order/signs from an existing iBOT color panel."""
    global LINEAR_COLOR_SCALE
    with Image.open(reference_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ys = np.floor((np.arange(grid[0]) + .5) * rgb.shape[0] / grid[0]).astype(int)
    xs = np.floor((np.arange(grid[1]) + .5) * rgb.shape[1] / grid[1]).astype(int)
    target = rgb[np.ix_(ys, xs)].reshape(-1, 3).astype(np.float64) / 255
    candidates = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            aligned = scores[:, permutation] * np.asarray(signs)
            gain = None
            if PCA_COLOR_PROTOCOL == "legacy_linear_l2":
                # Recover the one shared linear scale from unsaturated color values.
                mask = (target > 0) & (target < 1)
                denominator = np.sum(aligned[mask] ** 2)
                if denominator <= 0:
                    continue
                gain = np.sum(aligned[mask] * (target[mask] + .5 / 255 - .5)) / denominator
                if gain <= 0:
                    continue
                colors = np.clip(.5 + aligned * gain, 0, 1)
            else:
                colors = 1 / (1 + np.exp(-np.clip(SIGMOID_GAIN * aligned, -80, 80)))
            candidates.append((float(np.mean((colors - target) ** 2)), permutation, signs, gain))
    if not candidates:
        raise ValueError("Reference panel has no recoverable color scale")
    mse, permutation, signs, gain = min(candidates)
    if mse > 1e-4:
        raise ValueError(
            f"iBOT reference colors cannot be reproduced by PCA channel order/signs (RGB MSE={mse:.6g}); "
            "check the reference checkpoint, image and joint-view PCA protocol"
        )
    if gain is not None:
        LINEAR_COLOR_SCALE = float(1 / (2 * gain))
    return scores[:, permutation] * np.asarray(signs), {
        "target_components_zero_based": list(permutation), "signs": list(signs),
        "reference": True, "color_reference": str(reference_path), "reference_rgb_mse": mse,
        "linear_color_scale": LINEAR_COLOR_SCALE,
    }


def _color_variant(variant: int) -> dict:
    """48 signed RGB permutations at three shared contrast levels."""
    if type(variant) is not int or not 0 <= variant < 144:
        raise ValueError("Color variant must be an integer from 0 to 143")
    permutations = list(itertools.permutations(range(3)))
    signs = list(itertools.product((1, -1), repeat=3))
    return {
        "variant": variant,
        "rgb_components_zero_based": list(permutations[(variant % 48) // 8]),
        "rgb_signs": list(signs[variant % 8]),
        "sigmoid_gain": (1.5, 1.0, 2.0)[variant // 48],
        "shared_across_models": True,
    }


def _projected_to_rgb(projected, grid=None, size=None, color_variant=None) -> Image.Image:
    grid = grid or (math.isqrt(len(projected)),) * 2
    size = size or (VIS_RESOLUTION, VIS_RESOLUTION)
    gain = SIGMOID_GAIN
    if color_variant is not None:
        mapping = _color_variant(color_variant)
        projected = projected[:, mapping["rgb_components_zero_based"]] * np.asarray(mapping["rgb_signs"])
        gain = mapping["sigmoid_gain"]
    # Same whitening and color mapping for every model.
    if PCA_COLOR_PROTOCOL == "legacy_linear_l2":
        if LINEAR_COLOR_SCALE is None:
            raise ValueError("Legacy PCA requires a recovered reference color scale")
        contrast = gain / SIGMOID_GAIN if color_variant is not None else 1.0
        rgb = np.clip(.5 + contrast * projected / (2 * LINEAR_COLOR_SCALE), 0, 1)
    else:
        rgb = 1.0 / (1.0 + np.exp(-np.clip(gain * projected, -80, 80)))
    array = np.clip(rgb.reshape(*grid, 3) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).resize(size, MAP_RESAMPLING)


def _write_manifest(manifest):
    path = OUTPUT_DIR / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def _recolor_existing(source: Path, variant: int) -> None:
    """Render new colors from saved aligned PCA scores, without model inference."""
    global PCA_COLOR_PROTOCOL, LINEAR_COLOR_SCALE
    mapping = _color_variant(variant)
    source = source.expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest["images"]) != 1:
        raise ValueError("Recoloring requires a complete single-image PCA export")
    PCA_COLOR_PROTOCOL = manifest.get("pca_color_protocol", "sigmoid_whitened")
    if PCA_COLOR_PROTOCOL == "legacy_linear_l2":
        LINEAR_COLOR_SCALE = float(manifest["color_mapping"]["scale"])
        if not math.isfinite(LINEAR_COLOR_SCALE) or LINEAR_COLOR_SCALE <= 0:
            raise ValueError("Saved linear color scale must be positive and finite")
        mapping.update(type="clipped linear", scale=LINEAR_COLOR_SCALE * SIGMOID_GAIN / mapping["sigmoid_gain"])
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.update(status="running", recolored_from=str(source), color_mapping=mapping)
    _write_manifest(manifest)
    try:
        record = manifest["images"][0]
        shutil.copyfile(source / record["original"], OUTPUT_DIR / record["original"])
        if set(record["visualizations"]) != set(CHECKPOINTS):
            raise ValueError("Source export must contain all four models")
        for name, visualization in record["visualizations"].items():
            with np.load(source / visualization["scores"], allow_pickle=False) as archive:
                scores = archive["aligned_scores"]
                if scores.shape != (*visualization["grid"], 3) or not np.isfinite(scores).all():
                    raise ValueError(f"Invalid saved PCA scores for {name}")
                _projected_to_rgb(
                    scores.reshape(-1, 3), tuple(visualization["grid"]), tuple(record["size"]), variant
                ).save(OUTPUT_DIR / visualization["png"], dpi=(DPI, DPI))
            shutil.copyfile(source / visualization["scores"], OUTPUT_DIR / visualization["scores"])
            print(f"Variant {variant:03d}: {name}", flush=True)
        manifest["status"] = "complete"
        _write_manifest(manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        _write_manifest(manifest)
        raise
    print(f"Done: four PCA color panels in {OUTPUT_DIR}", flush=True)


def main() -> None:
    global INPUT_IMAGE, OUTPUT_DIR, COLOR_REFERENCE, PCA_COLOR_PROTOCOL, LINEAR_COLOR_SCALE
    global COLOR_MAPPING_SOURCE, N_LAST_LAYERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Process only this image with all four models")
    parser.add_argument("--output-dir", type=Path, help="Separate output folder; must be empty")
    parser.add_argument("--recolor-from", type=Path, help="Reuse a complete single-image export; runs on CPU")
    parser.add_argument("--color-variant", type=int, help="Shared RGB signs/order and contrast variant (0-143)")
    parser.add_argument("--match-imagenet", action="store_true", help="Match ImageNet visualization features and joint-view PCA for an exported --image")
    parser.add_argument("--imagenet-protocol", type=Path, help="Original ImageNet protocol.json to identify exact checkpoints and settings")
    parser.add_argument("--ibot-checkpoint", type=Path, help="Exact iBOT checkpoint used by the reference visualization")
    parser.add_argument("--color-reference", type=Path, help="Existing iBOT original-view PCA panel; recover matching channel order/signs")
    parser.add_argument("--color-mapping-from", type=Path, help="Reuse a completed legacy PCA export's recovered RGB order/signs and linear scale for a different image")
    parser.add_argument("--legacy-linear-pca", action="store_true", help="Reproduce older ImageNet figures: L2-normalized patches, unwhitened PCA and shared clipped linear colors")
    args = parser.parse_args()
    if args.color_mapping_from is not None:
        if not args.match_imagenet or args.color_reference is not None:
            parser.error("--color-mapping-from requires --match-imagenet and cannot be combined with --color-reference")
        source = args.color_mapping_from.expanduser().resolve()
        source_manifest = source / "manifest.json" if source.is_dir() else source
        previous = json.loads(source_manifest.read_text())
        if previous.get("status") != "complete" or previous.get("pca_color_protocol") != "legacy_linear_l2":
            parser.error("--color-mapping-from requires a complete legacy_linear_l2 export")
        if len(previous.get("images", [])) != 1:
            parser.error("--color-mapping-from requires a single-image export")
        mapping = previous["color_mapping"]
        LINEAR_COLOR_SCALE = float(mapping["scale"])
        if mapping.get("type") != "clipped linear" or not math.isfinite(LINEAR_COLOR_SCALE) or LINEAR_COLOR_SCALE <= 0:
            parser.error("The saved color mapping must have a finite positive clipped-linear scale")
        alignment = previous["images"][0]["visualizations"][PCA_REFERENCE_MODEL]["alignment"]
        permutation = alignment["target_components_zero_based"]
        signs = alignment["signs"]
        if sorted(permutation) != [0, 1, 2] or len(signs) != 3 or any(sign not in (-1, 1) for sign in signs):
            parser.error("The saved RGB permutation/signs are invalid")
        COLOR_MAPPING_SOURCE = {
            "manifest": str(source_manifest), "scale": LINEAR_COLOR_SCALE,
            "target_components_zero_based": permutation, "signs": signs,
            "original_reference_panel": mapping.get("reference_panel"),
        }
        PCA_COLOR_PROTOCOL = "legacy_linear_l2"
    if args.legacy_linear_pca:
        if not args.match_imagenet or (args.color_reference is None and args.color_mapping_from is None):
            parser.error("--legacy-linear-pca requires --match-imagenet and a color reference or saved color mapping")
        PCA_COLOR_PROTOCOL = "legacy_linear_l2"
    if args.color_reference is not None and not args.match_imagenet:
        parser.error("--color-reference requires --match-imagenet")
    if args.imagenet_protocol is not None and not args.match_imagenet:
        parser.error("--imagenet-protocol requires --match-imagenet")
    if args.match_imagenet and (args.image is None or args.recolor_from is not None):
        parser.error("--match-imagenet requires --image and cannot be combined with --recolor-from")
    if args.recolor_from is not None:
        if args.image is not None or args.color_variant is None:
            parser.error("--recolor-from requires --color-variant and cannot be combined with --image")
        OUTPUT_DIR = (args.output_dir.expanduser().resolve() if args.output_dir is not None else
                      args.recolor_from.expanduser().resolve().parent / f"pca_color_variant_{args.color_variant:03d}")
        _recolor_existing(args.recolor_from, args.color_variant)
        return
    if args.color_variant is not None:
        parser.error("--color-variant requires --recolor-from")
    if args.image is not None:
        INPUT_IMAGE = args.image.expanduser().resolve()
        OUTPUT_DIR = REPO_ROOT / "output" / f"pca_visualizations_{INPUT_IMAGE.stem}"
    if args.match_imagenet:
        _configure_imagenet_match(args.imagenet_protocol.expanduser().resolve() if args.imagenet_protocol is not None else None)
        OUTPUT_DIR = REPO_ROOT / "output" / f"pca_visualizations_{INPUT_IMAGE.stem}_imagenet_matched"
    if COLOR_MAPPING_SOURCE is not None:
        N_LAST_LAYERS = int(previous["last_layers_averaged"])
        IMAGENET_MATCH["implementation"].N_LAST_LAYERS = N_LAST_LAYERS
        for name in CHECKPOINTS:
            CHECKPOINTS[name] = Path(previous["models"][name]["checkpoint"])
    if args.ibot_checkpoint is not None:
        CHECKPOINTS["iBOT"] = args.ibot_checkpoint.expanduser().resolve()
    if args.color_reference is not None:
        COLOR_REFERENCE = args.color_reference.expanduser().resolve()
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.expanduser().resolve()
    _check_inputs()
    records = _sample_images()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {
        "status": "running", "seed": SEED, "images": records, "models": {},
        "pca_reference_model": PCA_REFERENCE_MODEL, "last_layers_averaged": N_LAST_LAYERS,
        "pca_color_protocol": PCA_COLOR_PROTOCOL,
        "pca": "per model and per image, top 3 PCs, whiten=True, full SVD",
        "alignment": "optimal permutation maximizing absolute Pearson correlations, then sign flips; no rotation or feature-space sharing",
        "alignment_grid": "reference patch centers; bilinear resampling of target scores only for correlations",
        "color_mapping": {"sigmoid_gain": SIGMOID_GAIN, "reference_signs": "positive skew"},
        "rendering": "native patch grids, nearest-neighbor upsampling, no spatial smoothing",
        "scope": "colors aligned across models within each image, not across different images",
        "selection": "one user-supplied image" if INPUT_IMAGE is not None else f"{N_IMAGES} unique ImageNet samples excluding the earlier cosine selection",
        "source_figure": None if INPUT_IMAGE is not None else "https://arxiv.org/html/2508.10104v1#S6.F13",
        "source_resolution": "native uploaded image; resize to common patch-size multiples" if INPUT_IMAGE is not None else "published example inputs are 768x576; resizing does not restore missing detail",
    }
    if IMAGENET_MATCH is not None:
        manifest.update(
            pca="per model and image, joint original/view_1/view_2 PCA, top 3 PCs, whiten=True, full SVD; export original scores",
            imagenet_match={key: value for key, value in IMAGENET_MATCH.items() if key != "implementation"},
            source_resolution="exported ImageNet original pixels at native resolution; no resizing",
        )
        manifest["color_mapping"]["reference_signs"] = "positive skew across all three views"
    if COLOR_REFERENCE is not None:
        manifest["color_mapping"].update(reference_panel=str(COLOR_REFERENCE), reference_signs="channel order and signs recovered from uploaded iBOT panel")
    if PCA_COLOR_PROTOCOL == "legacy_linear_l2":
        manifest["pca"] = "per model and image, L2-normalized patch features; joint original/view_1/view_2 unwhitened PCA, full SVD; export original scores"
    if COLOR_MAPPING_SOURCE is not None:
        manifest["color_mapping"] = {
            "type": "clipped linear", "scale": LINEAR_COLOR_SCALE,
            "shared_across_models": True, "source": COLOR_MAPPING_SOURCE,
        }
        manifest["scope"] = "reuse the saved RGB mapping; fit PCA separately on this image and align the other models to its iBOT scores"
    references = {}
    _write_manifest(manifest)
    print(f"Exporting {len(records)} images x 4 models to {OUTPUT_DIR}", flush=True)
    print(f"Features: last {N_LAST_LAYERS} normalized blocks; color reference: {PCA_REFERENCE_MODEL}", flush=True)
    try:
        for record in records:
            image = _prepare_image(record)
            filename = f"{record['name']}_original.png"
            image.save(OUTPUT_DIR / filename, dpi=(DPI, DPI))
            record.update(size=list(image.size), original=filename, visualizations={})
        _write_manifest(manifest)
        order = [PCA_REFERENCE_MODEL] + [name for name in CHECKPOINTS if name != PCA_REFERENCE_MODEL]
        for name in order:
            print(f"Loading {name}", flush=True)
            model, metadata = _load_model(name)
            if IMAGENET_MATCH is not None and IMAGENET_MATCH["protocol"] is not None and name in ("iBOT", "Ours"):
                expected = IMAGENET_MATCH["protocol"]["checkpoints"][name].get("checkpoint_fingerprint")
                if expected is not None and metadata.get("checkpoint_fingerprint") != expected:
                    raise ValueError(f"{name} checkpoint differs from the original ImageNet visualization weights")
            model = model.to(DEVICE).eval().requires_grad_(False)
            manifest["models"][name] = metadata
            for number, record in enumerate(records, 1):
                image = _prepare_image(record)
                grid = (image.height // PATCH_SIZES[name], image.width // PATCH_SIZES[name])
                if IMAGENET_MATCH is not None:
                    scores, variance = _fit_imagenet_pca(model, image, name)
                else:
                    features = _extract_dense_features(model, image, MODEL_FAMILIES[name], PATCH_SIZES[name])
                    scores, variance = _fit_pca(features)
                raw_scores = scores.copy()
                if name == PCA_REFERENCE_MODEL:
                    if IMAGENET_MATCH is None:
                        scores = _orient_component_signs(scores)
                    signs = np.where(np.sum(raw_scores * scores, axis=0) < 0, -1, 1)
                    alignment = {"target_components_zero_based": [0, 1, 2], "signs": signs.tolist(), "reference": True}
                    if COLOR_REFERENCE is not None:
                        scores, alignment = _match_reference_colors(scores, grid, COLOR_REFERENCE)
                        print(f"iBOT uploaded-color reference RGB MSE: {alignment['reference_rgb_mse']:.8g}", flush=True)
                        if PCA_COLOR_PROTOCOL == "legacy_linear_l2":
                            manifest["color_mapping"] = {
                                "type": "clipped linear", "scale": LINEAR_COLOR_SCALE,
                                "reference_panel": str(COLOR_REFERENCE), "shared_across_models": True,
                            }
                    elif COLOR_MAPPING_SOURCE is not None:
                        permutation = COLOR_MAPPING_SOURCE["target_components_zero_based"]
                        signs = COLOR_MAPPING_SOURCE["signs"]
                        scores = scores[:, permutation] * np.asarray(signs)
                        alignment = {
                            "target_components_zero_based": permutation, "signs": signs,
                            "reference": True, "color_mapping_source": COLOR_MAPPING_SOURCE["manifest"],
                            "linear_color_scale": LINEAR_COLOR_SCALE,
                        }
                    references[record["name"]] = (scores, grid)
                else:
                    reference, reference_grid = references[record["name"]]
                    scores, alignment = _align_scores(reference, reference_grid, scores, grid)
                prefix = f"{record['name']}_{name.lower()}"
                png_name = f"{prefix}_pca.png"
                array_name = f"{prefix}_pca.npz"
                _projected_to_rgb(scores, grid, image.size).save(OUTPUT_DIR / png_name, dpi=(DPI, DPI))
                np.savez_compressed(
                    OUTPUT_DIR / array_name, raw_scores=raw_scores.reshape(*grid, 3),
                    aligned_scores=scores.reshape(*grid, 3), explained_variance_ratio=variance,
                    permutation=np.array(alignment["target_components_zero_based"]),
                    signs=np.array(alignment["signs"]),
                )
                record["visualizations"][name] = {
                    "png": png_name, "scores": array_name, "grid": list(grid),
                    "explained_variance_ratio": variance.tolist(), "alignment": alignment,
                }
                _write_manifest(manifest)
                print(f"[{name}] {number:03d}/{len(records)} {record['name']} grid={grid}", flush=True)
            del model
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        manifest["status"] = "complete"
        _write_manifest(manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        _write_manifest(manifest)
        raise
    print(f"Done: {len(records) * 5} individual PNGs, {len(records) * 4} score archives, manifest.json", flush=True)


if __name__ == "__main__":
    main()
