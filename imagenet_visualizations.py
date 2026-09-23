#!/usr/bin/env python3
"""Standalone ImageNet patch-feature visualizations for iBOT and Ours.

Edit the constants below and run ``python imagenet_visualizations.py``. The
same sampled images, deterministic views, and patch coordinates are used for
both frozen teacher backbones. Each image produces similarity, PCA, region,
and cross-view correspondence figures in OUTPUT_DIR.
"""
from __future__ import annotations

import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
from torchvision import datasets, transforms as T

from evaluation.utils.common import load_backbone
from utils.pca_alignment import align_pca_components


# ---- Standalone configuration ---------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
IMAGENET_VAL = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/datasets/imagenet/val")
OUTPUT_DIR = REPO_ROOT / "output" / "imagenet_visualizations"
CHECKPOINTS = {
    "iBOT": REPO_ROOT / "checkpoints" / "ibot_vit_small.pth",
    "Ours": REPO_ROOT / "checkpoints" / "checkpoint_source1000_continuation0200.pth",
}
N_IMAGES = 5
SEED = 0
VIS_RESOLUTION = 560        # 35 x 35 patches for ViT-S/16
N_LAST_LAYERS = 4           # mean of normalized final block outputs
VIEW_CROP_FRACTION = 0.80  # opposing overlapping crops, each resized to 560
PCA_SIGMOID_GAIN = 1.5      # same whitened-PCA color mapping as pca_visualization.py
REGION_CLUSTERS = 6
KNN_NEIGHBORS = 12
SPATIAL_EDGE_WEIGHT = 0.20
TOP_K_CORRESPONDENCES = 32
DPI = 220
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# ---------------------------------------------------------------------------


GEOMETRIC_TRANSFORM = T.Compose([
    T.Resize(VIS_RESOLUTION, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop((VIS_RESOLUTION, VIS_RESOLUTION)),
])
MODEL_TRANSFORM = T.Compose([
    T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
VIEW_NAMES = ("original", "view_1", "view_2")
REGION_COLORS = (
    "#3B82F6", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6",
    "#EC4899", "#14B8A6", "#A855F7", "#84CC16", "#F97316",
)


@dataclass(frozen=True)
class PreparedImage:
    images: dict[str, Image.Image]
    boxes: dict[str, tuple[int, int, int, int]]
    dataset_index: int
    class_index: int
    class_name: str
    source_path: str
    patch_index: int


@dataclass(frozen=True)
class Tokens:
    cls: np.ndarray             # [D]
    patches: np.ndarray         # [patches, D]


def validate_configuration() -> None:
    if tuple(CHECKPOINTS) != ("iBOT", "Ours"):
        raise ValueError("CHECKPOINTS must contain iBOT and Ours in that order")
    if not IMAGENET_VAL.is_dir():
        raise FileNotFoundError(f"ImageNet validation ImageFolder is missing: {IMAGENET_VAL}")
    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing teacher checkpoints: {missing}")
    if N_IMAGES < 1 or VIS_RESOLUTION < 1 or not 0 < VIEW_CROP_FRACTION < 1:
        raise ValueError("N_IMAGES, VIS_RESOLUTION, and VIEW_CROP_FRACTION are invalid")
    if N_LAST_LAYERS < 1 or REGION_CLUSTERS < 2 or KNN_NEIGHBORS < 1:
        raise ValueError("N_LAST_LAYERS, REGION_CLUSTERS, and KNN_NEIGHBORS are invalid")
    if TOP_K_CORRESPONDENCES < 1 or REGION_CLUSTERS > len(REGION_COLORS):
        raise ValueError("TOP_K_CORRESPONDENCES or REGION_CLUSTERS is out of range")


def make_views(image: Image.Image) -> tuple[dict[str, Image.Image], dict[str, tuple[int, int, int, int]]]:
    """Construct two reproducible, overlapping geometric augmentations."""
    if image.size != (VIS_RESOLUTION, VIS_RESOLUTION):
        raise ValueError("View source must already be the square visualization crop")
    side = VIS_RESOLUTION
    crop_side = max(1, min(side - 1, round(side * VIEW_CROP_FRACTION)))
    boxes = {
        "original": (0, 0, side, side),
        "view_1": (0, 0, crop_side, crop_side),
        "view_2": (side - crop_side, side - crop_side, side, side),
    }
    images = {"original": image}
    for name in ("view_1", "view_2"):
        images[name] = image.crop(boxes[name]).resize(
            (side, side), Image.Resampling.BICUBIC
        )
    return images, boxes


def sample_images(patch_size: int) -> list[PreparedImage]:
    dataset = datasets.ImageFolder(IMAGENET_VAL)
    if N_IMAGES > len(dataset):
        raise ValueError(f"Requested {N_IMAGES} images from {len(dataset)} ImageNet files")
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=N_IMAGES, replace=False)
    grid = VIS_RESOLUTION // patch_size
    selected = []
    for index in indices:
        source_path, class_index = dataset.samples[int(index)]
        with Image.open(source_path) as source:
            base = GEOMETRIC_TRANSFORM(source.convert("RGB"))
        images, boxes = make_views(base)
        selected.append(PreparedImage(
            images, boxes, int(index), int(class_index),
            dataset.classes[class_index], str(Path(source_path).resolve()),
            int(rng.integers(0, grid * grid)),
        ))
    return selected


@torch.inference_mode()
def extract_tokens(model: torch.nn.Module, image: Image.Image, patch_size: int) -> Tokens:
    """Average the last normalized layers; the model API omits register tokens."""
    depth = model.get_num_layers()
    if type(N_LAST_LAYERS) is not int or not 1 <= N_LAST_LAYERS <= depth:
        raise ValueError(f"N_LAST_LAYERS must be in [1, {depth}]")
    tensor = MODEL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    layers = model.get_intermediate_layers(tensor, n=N_LAST_LAYERS)
    if len(layers) != N_LAST_LAYERS:
        raise ValueError("Backbone returned an unexpected number of intermediate layers")
    expected = (VIS_RESOLUTION // patch_size) ** 2
    for layer in layers:
        if layer.ndim != 3 or layer.shape[0] != 1 or layer.shape[1] != expected + 1:
            raise ValueError(
                "Intermediate layers must contain CLS and spatial patches only; "
                f"expected {expected + 1} tokens"
            )
    averaged = torch.stack([layer.float() for layer in layers]).mean(0)[0]
    return Tokens(averaged[0].cpu().numpy(), averaged[1:].cpu().numpy())


def cosine_similarity_to_patches(query: np.ndarray, patches: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    patches = np.asarray(patches, dtype=np.float64)
    if query.ndim != 1 or patches.ndim != 2 or query.shape[0] != patches.shape[1]:
        raise ValueError("Query and patch features have incompatible dimensions")
    query_norm = np.linalg.norm(query)
    patch_norms = np.linalg.norm(patches, axis=1)
    if query_norm <= 0 or np.any(patch_norms <= 0):
        raise ValueError("Cosine similarity requires nonzero token features")
    return (patches @ query) / (patch_norms * query_norm)


def pca_triplet(tokens: dict[str, Tokens]) -> dict[str, np.ndarray]:
    """Fit one whitened PCA across original plus both independently encoded views."""
    lengths = [len(tokens[name].patches) for name in VIEW_NAMES]
    features = np.concatenate([tokens[name].patches for name in VIEW_NAMES])
    if features.shape[0] < 4 or features.shape[1] < 3:
        raise ValueError("At least three feature dimensions and four patches are needed for PCA")
    projected = PCA(n_components=3, whiten=True, svd_solver="full").fit_transform(features)
    # Reference sign convention used by pca_visualization.py.
    for component in range(3):
        column = projected[:, component]
        second_moment = np.mean(column ** 2)
        if second_moment > 0 and np.mean(column ** 3) / second_moment ** 1.5 < 0:
            projected[:, component] *= -1
    offsets = np.cumsum([0, *lengths])
    return {
        name: projected[offsets[index]:offsets[index + 1]].astype(np.float32)
        for index, name in enumerate(VIEW_NAMES)
    }


def align_pca_triplet(reference: dict[str, np.ndarray], target: dict[str, np.ndarray]):
    alignment = align_pca_components(reference["original"], target["original"])
    aligned = {
        name: target[name][:, alignment.permutation] * alignment.signs
        for name in VIEW_NAMES
    }
    record = {
        "component_permutation": alignment.permutation.tolist(),
        "component_signs": alignment.signs.tolist(),
        "correlation_matrix": alignment.correlations.tolist(),
    }
    return aligned, record


def pca_to_rgb(scores: np.ndarray, grid: int) -> Image.Image:
    if scores.shape != (grid * grid, 3):
        raise ValueError("PCA scores do not match the patch grid")
    rgb = 1.0 / (1.0 + np.exp(-PCA_SIGMOID_GAIN * scores))
    pixels = np.clip(rgb.reshape(grid, grid, 3) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, "RGB").resize(
        (VIS_RESOLUTION, VIS_RESOLUTION), Image.Resampling.NEAREST
    )


def infer_knn_regions(tokens: dict[str, Tokens], grid: int, seed: int) -> dict[str, np.ndarray]:
    """Spectral partition of a shared feature k-NN graph over the three views."""
    features = np.concatenate([tokens[name].patches for name in VIEW_NAMES]).astype(np.float64)
    count = len(features)
    if count <= max(REGION_CLUSTERS, KNN_NEIGHBORS):
        raise ValueError("Too few patch features for k-NN region clustering")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    neighbors = NearestNeighbors(n_neighbors=KNN_NEIGHBORS + 1, metric="cosine")
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors(features)
    rows, columns, values = [], [], []
    for row in range(count):
        for distance, column in zip(distances[row], indices[row]):
            if row == column:
                continue
            rows.append(row)
            columns.append(int(column))
            values.append(float(np.exp(-max(0.0, distance))))
    for view_index in range(len(VIEW_NAMES)):
        offset = view_index * grid * grid
        for y in range(grid):
            for x in range(grid):
                source = offset + y * grid + x
                for neighbor in ((y + 1, x), (y, x + 1)):
                    ny, nx = neighbor
                    if ny < grid and nx < grid:
                        target = offset + ny * grid + nx
                        rows.extend((source, target))
                        columns.extend((target, source))
                        values.extend((SPATIAL_EDGE_WEIGHT, SPATIAL_EDGE_WEIGHT))
    graph = coo_matrix((values, (rows, columns)), shape=(count, count)).tocsr()
    graph = graph.maximum(graph.T)
    labels = SpectralClustering(
        n_clusters=REGION_CLUSTERS, affinity="precomputed",
        assign_labels="kmeans", random_state=seed, n_init=10,
    ).fit_predict(graph)
    patch_count = grid * grid
    return {
        name: labels[index * patch_count:(index + 1) * patch_count].copy()
        for index, name in enumerate(VIEW_NAMES)
    }


def align_region_labels(reference: np.ndarray, target: dict[str, np.ndarray]):
    """Assign Ours' arbitrary cluster colors by overlap on the original grid."""
    overlap = np.zeros((REGION_CLUSTERS, REGION_CLUSTERS), dtype=np.int64)
    for reference_label, target_label in zip(reference, target["original"]):
        overlap[int(reference_label), int(target_label)] += 1
    rows, columns = linear_sum_assignment(-overlap)
    mapping = np.empty(REGION_CLUSTERS, dtype=np.int64)
    mapping[columns] = rows
    return {name: mapping[target[name]] for name in VIEW_NAMES}, mapping.tolist()


def patch_centers_in_source(box: tuple[int, int, int, int], grid: int) -> np.ndarray:
    left, top, right, bottom = box
    locations = np.indices((grid, grid)).reshape(2, -1).T
    x = left + (locations[:, 1] + .5) * (right - left) / grid
    y = top + (locations[:, 0] + .5) * (bottom - top) / grid
    return np.stack((x, y), axis=1)


def top_correspondences(
    source: np.ndarray, target: np.ndarray,
    source_box: tuple[int, int, int, int], target_box: tuple[int, int, int, int],
    source_regions: np.ndarray, target_regions: np.ndarray,
    grid: int, top_k: int = TOP_K_CORRESPONDENCES,
) -> list[dict]:
    """Top mutual nearest patches inside the two views' geometric overlap."""
    if source.shape != target.shape or source.shape[0] != grid * grid:
        raise ValueError("Correspondence features must share a square patch grid")
    overlap = (
        max(source_box[0], target_box[0]), max(source_box[1], target_box[1]),
        min(source_box[2], target_box[2]), min(source_box[3], target_box[3]),
    )
    if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
        raise ValueError("Augmented views must overlap")
    def eligible(box):
        centers = patch_centers_in_source(box, grid)
        return np.flatnonzero(
            (centers[:, 0] >= overlap[0]) & (centers[:, 0] < overlap[2]) &
            (centers[:, 1] >= overlap[1]) & (centers[:, 1] < overlap[3])
        )
    source_indices, target_indices = eligible(source_box), eligible(target_box)
    if len(source_indices) == 0 or len(target_indices) == 0:
        raise ValueError("No patch centers fall inside the augmented-view overlap")
    a = source[source_indices].astype(np.float64)
    b = target[target_indices].astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    similarities = a @ b.T
    best_targets = similarities.argmax(axis=1)
    best_sources = similarities.argmax(axis=0)
    matches = []
    for source_local, target_local in enumerate(best_targets):
        if best_sources[target_local] != source_local:
            continue
        source_index = int(source_indices[source_local])
        target_index = int(target_indices[target_local])
        matches.append({
            "source_patch": source_index, "target_patch": target_index,
            "cosine": float(similarities[source_local, target_local]),
            "source_region": int(source_regions[source_index]),
            "target_region": int(target_regions[target_index]),
            "same_region": bool(source_regions[source_index] == target_regions[target_index]),
        })
    return sorted(matches, key=lambda item: -item["cosine"])[:top_k]


def _heatmap_axis(axis, image: Image.Image, scores: np.ndarray, grid: int, title: str):
    image_array = np.asarray(image)
    heat = scores.reshape(grid, grid)
    axis.imshow(image_array)
    axis.imshow(heat, cmap="magma", vmin=-1, vmax=1, alpha=.68,
                interpolation="nearest", extent=(0, VIS_RESOLUTION, VIS_RESOLUTION, 0))
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def render_similarity(
    prepared: PreparedImage, all_tokens: dict[str, dict[str, Tokens]],
    grid: int, path: Path, kind: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for row, model_name in enumerate(CHECKPOINTS):
        tokens = all_tokens[model_name]["original"]
        query = tokens.cls if kind == "cls" else tokens.patches[prepared.patch_index]
        similarities = cosine_similarity_to_patches(query, tokens.patches)
        axes[row, 0].imshow(prepared.images["original"])
        if kind == "patch":
            y, x = divmod(prepared.patch_index, grid)
            axes[row, 0].scatter([(x + .5) * VIS_RESOLUTION / grid],
                                 [(y + .5) * VIS_RESOLUTION / grid],
                                 marker="x", s=130, c="white", linewidths=3)
        axes[row, 0].set_title(f"{model_name}: query", fontsize=10)
        axes[row, 0].axis("off")
        _heatmap_axis(axes[row, 1], prepared.images["original"], similarities,
                      grid, f"{model_name}: cosine similarity")
    fig.suptitle("CLS to patch similarity" if kind == "cls" else
                 f"Patch {prepared.patch_index} to all patches", fontsize=13)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_pca(prepared: PreparedImage, pca_maps: dict[str, dict[str, Image.Image]], path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    for row, model_name in enumerate(CHECKPOINTS):
        for column, (title, image) in enumerate((
            ("Input image", prepared.images["original"]),
            ("PCA: original", pca_maps[model_name]["original"]),
            ("PCA: view 1", pca_maps[model_name]["view_1"]),
            ("PCA: view 2", pca_maps[model_name]["view_2"]),
        )):
            axes[row, column].imshow(image)
            axes[row, column].set_title(f"{model_name} | {title}", fontsize=10)
            axes[row, column].axis("off")
    fig.suptitle("One joint PCA per model and image; shared RGB mapping across all three views", fontsize=12)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_regions(prepared: PreparedImage, regions: dict[str, dict[str, np.ndarray]],
                   grid: int, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    cmap = ListedColormap(REGION_COLORS[:REGION_CLUSTERS])
    for row, model_name in enumerate(CHECKPOINTS):
        for column, view_name in enumerate(VIEW_NAMES):
            axes[row, column].imshow(prepared.images[view_name])
            axes[row, column].imshow(
                regions[model_name][view_name].reshape(grid, grid),
                cmap=cmap, vmin=-.5, vmax=REGION_CLUSTERS - .5,
                interpolation="nearest", alpha=.65,
                extent=(0, VIS_RESOLUTION, VIS_RESOLUTION, 0),
            )
            axes[row, column].set_title(f"{model_name} | {view_name}", fontsize=10)
            axes[row, column].axis("off")
    fig.suptitle(f"Feature k-NN graph regions ({REGION_CLUSTERS} clusters)", fontsize=12)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_correspondences(prepared: PreparedImage, matches: dict[str, list[dict]],
                           grid: int, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    gap = 35
    offset = VIS_RESOLUTION + gap
    for row, model_name in enumerate(CHECKPOINTS):
        axis = axes[row]
        axis.imshow(prepared.images["view_1"], extent=(0, VIS_RESOLUTION, VIS_RESOLUTION, 0))
        axis.imshow(prepared.images["view_2"], extent=(offset, offset + VIS_RESOLUTION, VIS_RESOLUTION, 0))
        for match in matches[model_name]:
            sy, sx = divmod(match["source_patch"], grid)
            ty, tx = divmod(match["target_patch"], grid)
            x1, y1 = (sx + .5) * VIS_RESOLUTION / grid, (sy + .5) * VIS_RESOLUTION / grid
            x2, y2 = offset + (tx + .5) * VIS_RESOLUTION / grid, (ty + .5) * VIS_RESOLUTION / grid
            color = "#22C55E" if match["same_region"] else "#EF4444"
            axis.plot((x1, x2), (y1, y2), color=color, alpha=.75, linewidth=1.2)
            axis.scatter((x1, x2), (y1, y2), c=color, s=13)
        count = len(matches[model_name])
        same = sum(match["same_region"] for match in matches[model_name])
        axis.set_xlim(0, offset + VIS_RESOLUTION)
        axis.set_ylim(VIS_RESOLUTION, 0)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(f"{model_name} | {same}/{count} mutual top matches share an inferred region", fontsize=11)
    fig.suptitle("Cross-view patch correspondence: green = same region, red = different region", fontsize=12)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"Cannot write an empty manifest: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prepared_images = None
    all_features: dict[str, list[dict[str, Tokens]]] = {}
    checkpoint_metadata = {}
    patch_size = None
    for model_name, path in CHECKPOINTS.items():
        model, metadata = load_backbone(path, "teacher", "vit_small")
        loaded_patch_size = int(metadata["patch_size"])
        if VIS_RESOLUTION % loaded_patch_size:
            raise ValueError("VIS_RESOLUTION must be a multiple of the checkpoint patch size")
        if patch_size is None:
            patch_size = loaded_patch_size
            prepared_images = sample_images(patch_size)
        elif loaded_patch_size != patch_size:
            raise ValueError("The two checkpoints have different patch sizes")
        model = model.to(DEVICE).eval()
        checkpoint_metadata[model_name] = metadata
        all_features[model_name] = []
        print(f"Extracting {model_name} features from {len(prepared_images)} images", flush=True)
        for number, prepared in enumerate(prepared_images, 1):
            all_features[model_name].append({
                name: extract_tokens(model, prepared.images[name], patch_size)
                for name in VIEW_NAMES
            })
            print(f"  {number}/{len(prepared_images)}", flush=True)
        model.cpu()
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    grid = VIS_RESOLUTION // patch_size
    write_csv(OUTPUT_DIR / "selected_images.csv", [{
        "image_number": number, "dataset_index": prepared.dataset_index,
        "class_index": prepared.class_index, "class_name": prepared.class_name,
        "source_path": prepared.source_path, "selected_patch_index": prepared.patch_index,
    } for number, prepared in enumerate(prepared_images)])
    alignment_records, correspondence_rows = [], []
    for number, prepared in enumerate(prepared_images):
        prefix = f"image_{number:03d}"
        prepared.images["original"].save(OUTPUT_DIR / f"{prefix}_original.png")
        tokens = {name: all_features[name][number] for name in CHECKPOINTS}
        pca_scores = {name: pca_triplet(tokens[name]) for name in CHECKPOINTS}
        pca_scores["Ours"], alignment = align_pca_triplet(pca_scores["iBOT"], pca_scores["Ours"])
        alignment_records.append({"image_number": number, **alignment})
        pca_maps = {
            model: {view: pca_to_rgb(pca_scores[model][view], grid) for view in VIEW_NAMES}
            for model in CHECKPOINTS
        }
        regions = {
            model: infer_knn_regions(tokens[model], grid, SEED + number)
            for model in CHECKPOINTS
        }
        regions["Ours"], mapping = align_region_labels(regions["iBOT"]["original"], regions["Ours"])
        alignment_records[-1]["region_color_mapping"] = mapping
        matches = {
            model: top_correspondences(
                tokens[model]["view_1"].patches, tokens[model]["view_2"].patches,
                prepared.boxes["view_1"], prepared.boxes["view_2"],
                regions[model]["view_1"], regions[model]["view_2"], grid,
            ) for model in CHECKPOINTS
        }
        for model in CHECKPOINTS:
            for rank, match in enumerate(matches[model], 1):
                correspondence_rows.append({
                    "image_number": number, "model": model, "rank": rank, **match,
                })
        render_similarity(prepared, tokens, grid, OUTPUT_DIR / f"{prefix}_cls_similarity.png", "cls")
        render_similarity(prepared, tokens, grid, OUTPUT_DIR / f"{prefix}_patch_similarity.png", "patch")
        render_pca(prepared, pca_maps, OUTPUT_DIR / f"{prefix}_pca_views.png")
        render_regions(prepared, regions, grid, OUTPUT_DIR / f"{prefix}_knn_regions.png")
        render_correspondences(prepared, matches, grid, OUTPUT_DIR / f"{prefix}_correspondences.png")
        print(f"Rendered {prefix} ({prepared.class_name})", flush=True)
    if correspondence_rows:
        write_csv(OUTPUT_DIR / "correspondences.csv", correspondence_rows)
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps({
        "checkpoints": checkpoint_metadata,
        "imagenet_val": str(IMAGENET_VAL.resolve()), "images": N_IMAGES,
        "seed": SEED, "resolution": VIS_RESOLUTION, "patch_size": patch_size,
        "feature": f"mean of last {N_LAST_LAYERS} normalized transformer blocks",
        "views": "top-left and bottom-right overlapping crops of the same square image, resized independently",
        "view_crop_fraction": VIEW_CROP_FRACTION,
        "pca": "joint whitened PCA across original and two views per model; Ours' components aligned to iBOT on original patch grid; shared sigmoid RGB mapping",
        "region_inference": "spectral clustering of joint feature cosine k-NN graph with spatial neighbor edges",
        "region_clusters": REGION_CLUSTERS, "knn_neighbors": KNN_NEIGHBORS,
        "correspondence": "top mutual cosine-nearest patches in the geometric overlap of two views",
        "top_k_correspondences": TOP_K_CORRESPONDENCES,
        "alignments": alignment_records,
    }, indent=2) + "\n")
    print(f"Wrote figures and manifests to {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
