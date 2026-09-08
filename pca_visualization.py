#!/usr/bin/env python3


from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from torchvision import datasets, transforms as T


# -----------------------------------------------------------------------------
# Repository paths and experiment settings (hard-coded by design)
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils.common import load_backbone  # noqa: E402

IMAGENET_VAL = REPO_ROOT / "dataset" / "imagenet" / "val"
OUTPUT_DIR = REPO_ROOT / "output" / "pca_visualizations"

CHECKPOINTS = {
    "Official iBOT": REPO_ROOT / "checkpoints" / "ibot_vit_small.pth",
    "200 epoch control": (
        REPO_ROOT
        / "output"
        / "region_loss"
        / "20260903T181757Z-slurm-1309714"
        / "checkpoint_source1000_continuation0200.pth"
    ),
    "Overlap": (
        REPO_ROOT
        / "output"
        / "region_loss"
        / "20260904T062926Z-slurm-1309971"
        / "checkpoint_source1000_continuation0200.pth"
    ),
}

CHECKPOINT_KEY = "teacher"
ARCH = "vit_small"

N_IMAGES = 100
SEED = 0

# 560 is divisible by the ViT-S/16 patch size and gives a 35 x 35 dense grid.
# Increase to 1120 for a CRISP-like high-resolution figure if GPU memory/time
# are not a concern (70 x 70 patch tokens, much more expensive attention).
VIS_RESOLUTION = 560

# Average corresponding patch embeddings from the last N transformer blocks.
# Each block output is LayerNorm-normalized. Set to 1 for the old visualization.
N_LAST_LAYERS = 4

# PCA is whitened so components have comparable scale. The sigmoid provides a
# smooth [0, 1] color mapping without per-image min/max clipping.
SIGMOID_GAIN = 1.5

DEVICE = torch.device("cuda")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Image transforms
# -----------------------------------------------------------------------------
GEOMETRIC_TRANSFORM = T.Compose(
    [
        T.Resize(VIS_RESOLUTION, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop((VIS_RESOLUTION, VIS_RESOLUTION)),
    ]
)

MODEL_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def _check_inputs() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this visualization script.")

    if not IMAGENET_VAL.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {IMAGENET_VAL}")

    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The following checkpoints are missing:\n  - " + "\n  - ".join(missing)
        )


def _sample_images() -> tuple[datasets.ImageFolder, list[int]]:
    """Select the same deterministic ImageNet images for all checkpoints."""
    dataset = datasets.ImageFolder(IMAGENET_VAL)
    if N_IMAGES > len(dataset):
        raise ValueError(f"N_IMAGES={N_IMAGES} exceeds dataset size {len(dataset)}")

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=N_IMAGES, replace=False).tolist()
    return dataset, indices


def _prepare_selected_images(
    dataset: datasets.ImageFolder, indices: list[int]
) -> tuple[list[Image.Image], list[str]]:
    """Load and geometrically preprocess the selected images once."""
    images: list[Image.Image] = []
    descriptions: list[str] = []

    lines = ["index\tclass_index\tclass_name\tpath"]
    for index in indices:
        image, class_index = dataset[index]
        image = image.convert("RGB")
        image = GEOMETRIC_TRANSFORM(image)

        class_name = dataset.classes[class_index]
        sample_path = Path(dataset.samples[index][0])

        images.append(image)
        descriptions.append(f"{class_name} | idx={index}")
        lines.append(f"{index}\t{class_index}\t{class_name}\t{sample_path}")

    (OUTPUT_DIR / "selected_images.tsv").write_text("\n".join(lines) + "\n")
    return images, descriptions


@torch.inference_mode()
def _extract_dense_features(model: torch.nn.Module, image: Image.Image) -> np.ndarray:
    """Average the last N normalized block outputs at each patch position."""
    depth = model.get_num_layers()
    if type(N_LAST_LAYERS) is not int or not 1 <= N_LAST_LAYERS <= depth:
        raise ValueError(
            f"N_LAST_LAYERS must be an integer between 1 and {depth}, "
            f"got {N_LAST_LAYERS!r}"
        )

    tensor = MODEL_TRANSFORM(image).unsqueeze(0).to(DEVICE, non_blocking=True)

    # Exclude CLS from every block and average in FP32 before fitting PCA.
    # Unlike the dense evaluator (last block only), this mixes multiple depths.
    layers = model.get_intermediate_layers(tensor, n=N_LAST_LAYERS)
    tokens = torch.stack([layer[:, 1:].float() for layer in layers], dim=0).mean(dim=0)
    tokens = tokens.squeeze(0).cpu().numpy()

    grid_size = math.isqrt(tokens.shape[0])
    if grid_size * grid_size != tokens.shape[0]:
        raise ValueError(f"Non-square patch-token grid with {tokens.shape[0]} tokens")

    return tokens


def _orient_component_signs(projected: np.ndarray) -> np.ndarray:
    """Resolve PCA sign ambiguity using the positive-skew convention used by CAPI."""
    projected = projected.copy()
    for component_index in range(projected.shape[1]):
        component = projected[:, component_index]
        second_moment = np.mean(component**2)
        if second_moment <= 0:
            continue
        skewness = np.mean(component**3) / (second_moment ** 1.5)
        if skewness < 0:
            projected[:, component_index] *= -1.0
    return projected


def _pca_to_rgb(features: np.ndarray) -> Image.Image:
    """Fit PCA for one image and convert the first 3 components into an RGB map."""
    pca = PCA(n_components=3, whiten=True, svd_solver="full")
    projected = pca.fit_transform(features).astype(np.float32, copy=False)
    projected = _orient_component_signs(projected)

    # Smooth contrast mapping. Whitened PCs are approximately unit scale, so
    # sigmoid(gain * x) gives a stable mapping without hard min/max saturation.
    rgb = 1.0 / (1.0 + np.exp(-SIGMOID_GAIN * projected))

    grid_size = math.isqrt(rgb.shape[0])
    rgb = rgb.reshape(grid_size, grid_size, 3)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    # Upsample the patch grid for visualization only.
    return Image.fromarray(rgb, mode="RGB").resize(
        (VIS_RESOLUTION, VIS_RESOLUTION),
        resample=Image.Resampling.BICUBIC,
    )


def _save_per_image_comparison(
    image_index: int,
    original: Image.Image,
    maps: dict[str, Image.Image],
) -> None:
    """Save original + all model PCA maps for a single sampled image."""
    columns = [("Original", original)] + list(maps.items())
    header_height = 34
    width = VIS_RESOLUTION * len(columns)
    height = VIS_RESOLUTION + header_height

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for column, (label, image) in enumerate(columns):
        x = column * VIS_RESOLUTION
        canvas.paste(image, (x, header_height))
        draw.text((x + 8, 9), label, fill="black")

    canvas.save(OUTPUT_DIR / f"image_{image_index:02d}_comparison.png")


def _save_global_grid(
    originals: list[Image.Image],
    all_maps: dict[str, list[Image.Image]],
) -> None:
    """Save a paper-style grid: rows are images, columns are models."""
    # Keep the global montage compact while individual images remain 560 x 560.
    tile_size = 224
    header_height = 34
    row_label_width = 42

    labels = ["Original"] + list(all_maps.keys())
    columns = len(labels)
    rows = len(originals)

    width = row_label_width + columns * tile_size
    height = header_height + rows * tile_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for column, label in enumerate(labels):
        draw.text(
            (row_label_width + column * tile_size + 6, 9),
            label,
            fill="black",
        )

    for row in range(rows):
        draw.text((6, header_height + row * tile_size + 8), f"{row:02d}", fill="black")

        row_images = [originals[row]] + [all_maps[name][row] for name in all_maps]
        for column, image in enumerate(row_images):
            tile = image.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
            x = row_label_width + column * tile_size
            y = header_height + row * tile_size
            canvas.paste(tile, (x, y))

    canvas.save(OUTPUT_DIR / "comparison_grid.png")


def main() -> None:
    _check_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset, selected_indices = _sample_images()
    originals, descriptions = _prepare_selected_images(dataset, selected_indices)

    print(f"Selected {len(originals)} ImageNet validation images")
    print(f"Visualization resolution: {VIS_RESOLUTION} x {VIS_RESOLUTION}")
    print(f"PCA features: mean of the last {N_LAST_LAYERS} normalized block outputs")
    print(f"Output directory: {OUTPUT_DIR}")

    all_maps: dict[str, list[Image.Image]] = {}

    # Load one checkpoint at a time so GPU memory use stays small.
    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nLoading {model_name}: {checkpoint_path}", flush=True)
        model, metadata = load_backbone(checkpoint_path, CHECKPOINT_KEY, ARCH)
        model = model.to(DEVICE).eval()

        print(f"Loaded metadata: {metadata}", flush=True)
        model_maps: list[Image.Image] = []

        for image_number, image in enumerate(originals):
            features = _extract_dense_features(model, image)
            pca_image = _pca_to_rgb(features)
            model_maps.append(pca_image)

            safe_name = (
                model_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("+", "plus")
            )
            pca_image.save(
                OUTPUT_DIR / f"image_{image_number:02d}_{safe_name}_pca.png"
            )
            print(
                f"[{model_name}] {image_number + 1:02d}/{len(originals):02d} "
                f"{descriptions[image_number]} | patches={features.shape[0]}",
                flush=True,
            )

        all_maps[model_name] = model_maps

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save originals and per-image side-by-side comparisons.
    for image_number, original in enumerate(originals):
        original.save(OUTPUT_DIR / f"image_{image_number:02d}_original.png")
        maps = {name: images[image_number] for name, images in all_maps.items()}
        _save_per_image_comparison(image_number, original, maps)

    _save_global_grid(originals, all_maps)

    print("\nDone.")
    print(f"Main comparison: {OUTPUT_DIR / 'comparison_grid.png'}")
    print(f"Selected-image manifest: {OUTPUT_DIR / 'selected_images.tsv'}")


if __name__ == "__main__":
    main()
