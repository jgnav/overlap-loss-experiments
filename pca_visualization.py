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

import gc
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
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
OUTPUT_DIR = REPO_ROOT / "output/pca_visualizations"
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
N_IMAGES = 100
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
    """Six published examples plus 100 new, reproducible ImageNet images."""
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
    if record["source"] == "dinov3_figure13":
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


def _projected_to_rgb(projected, grid=None, size=None) -> Image.Image:
    grid = grid or (math.isqrt(len(projected)),) * 2
    size = size or (VIS_RESOLUTION, VIS_RESOLUTION)
    # Same whitening and sigmoid gain as before; no model-specific tuning.
    rgb = 1.0 / (1.0 + np.exp(-np.clip(SIGMOID_GAIN * projected, -80, 80)))
    array = np.clip(rgb.reshape(*grid, 3) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).resize(size, MAP_RESAMPLING)


def _write_manifest(manifest):
    path = OUTPUT_DIR / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    _check_inputs()
    records = _sample_images()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {
        "status": "running", "seed": SEED, "images": records, "models": {},
        "pca_reference_model": PCA_REFERENCE_MODEL, "last_layers_averaged": N_LAST_LAYERS,
        "pca": "per model and per image, top 3 PCs, whiten=True, full SVD",
        "alignment": "optimal permutation maximizing absolute Pearson correlations, then sign flips; no rotation or feature-space sharing",
        "alignment_grid": "reference patch centers; bilinear resampling of target scores only for correlations",
        "color_mapping": {"sigmoid_gain": SIGMOID_GAIN, "reference_signs": "positive skew"},
        "rendering": "native patch grids, nearest-neighbor upsampling, no spatial smoothing",
        "scope": "colors aligned across models within each image, not across different images",
        "selection": "100 unique ImageNet samples excluding the earlier cosine selection",
        "source_figure": "https://arxiv.org/html/2508.10104v1#S6.F13",
        "source_resolution": "published example inputs are 768x576; resizing does not restore missing detail",
    }
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
            model = model.to(DEVICE).eval().requires_grad_(False)
            manifest["models"][name] = metadata
            for number, record in enumerate(records, 1):
                image = _prepare_image(record)
                features = _extract_dense_features(model, image, MODEL_FAMILIES[name], PATCH_SIZES[name])
                grid = (image.height // PATCH_SIZES[name], image.width // PATCH_SIZES[name])
                scores, variance = _fit_pca(features)
                raw_scores = scores.copy()
                if name == PCA_REFERENCE_MODEL:
                    scores = _orient_component_signs(scores)
                    references[record["name"]] = (scores, grid)
                    signs = np.where(np.sum(raw_scores * scores, axis=0) < 0, -1, 1)
                    alignment = {"target_components_zero_based": [0, 1, 2], "signs": signs.tolist(), "reference": True}
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
