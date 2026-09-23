#!/usr/bin/env python3
"""Part-to-whole compositional re-encoding experiment on COCO val2017.

Edit the configuration below, then run this file directly. A parent object crop
and each of its non-overlapping parts are encoded in separate forward passes.
The experiment compares a matched continued-training control with the
region-consistency-trained model.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from evaluation.utils.common import load_backbone
from patch_concept_visualization import (
    _build_teacher_head, _checkpoint_argument, _num_special_tokens,
    _teacher_head_state, _teacher_state, _torch_load,
)


# ---- Standalone experiment configuration ---------------------------------
CHECKPOINTS = {
    "Matched control (+200 epochs)": Path("checkpoints/ibot_vit_small.pth"),
    "Region-trained (+200 epochs)": Path("checkpoints/checkpoint_source1000_continuation0200.pth"),
}
DATASETS_ROOT = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/datasets")
COCO_SPLIT = "val2017"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "composition_reencoding"
DEVICE = "cuda"
INPUT_SIZE = 224
PART_COUNTS = (2, 4, 9)
NUM_OBJECTS = 500
MIN_INSTANCE_AREA = 0.03
MAX_INSTANCE_AREA = 0.80
PARENT_PADDING = 0.15
MIN_PARENT_SIDE = 48
TEMPERATURE = 0.1
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
SEED = 0
DPI = 300
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CocoRegion:
    image_id: int
    annotation_id: int
    category_id: int
    category: str
    image_path: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Measurement:
    model: str
    image_id: int
    annotation_id: int
    category_id: int
    parts: int
    composition_js: float


def _decode_coco_segmentation(annotation, height, width, mask_utils):
    """Decode polygon, compressed RLE, or uncompressed RLE COCO masks."""
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    result = mask_utils.decode(rle).astype(bool)
    if result.ndim == 3:
        result = result.any(axis=2)
    if result.shape != (height, width):
        raise ValueError("Decoded COCO mask has incorrect dimensions")
    return result


def _coco_root(datasets_root):
    root = Path(datasets_root).expanduser().resolve()
    coco = root if root.name.casefold() == "coco" else root / "coco"
    if not coco.is_dir():
        raise FileNotFoundError(f"COCO directory not found: {coco}")
    return coco


def _resolve_coco_image(coco_root, split, filename):
    candidates = (
        coco_root / "images" / split / Path(filename).name,
        coco_root / split / Path(filename).name,
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one COCO image for {filename}; found {matches}"
        )
    return matches[0]


def load_coco_regions(
    datasets_root=DATASETS_ROOT,
    split=COCO_SPLIT,
    num_objects=NUM_OBJECTS,
    seed=SEED,
):
    """Sample valid object instances and derive their padded parent crops."""
    if split != "val2017":
        raise ValueError("This held-out experiment requires COCO val2017")
    if num_objects < 2:
        raise ValueError("NUM_OBJECTS must be at least two")
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("COCO mask loading requires pycocotools") from exc

    coco = _coco_root(datasets_root)
    annotation_path = coco / "annotations" / f"instances_{split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    with annotation_path.open() as stream:
        data = json.load(stream)
    images = {int(row["id"]): row for row in data.get("images", [])}
    categories = {int(row["id"]): row["name"] for row in data.get("categories", [])}
    candidates = []
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        record = images.get(image_id)
        if record is None or annotation.get("iscrowd", 0) or not annotation.get("segmentation"):
            continue
        image_area = float(record["height"] * record["width"])
        area_fraction = float(annotation.get("area", 0)) / image_area
        _, _, width, height = map(float, annotation.get("bbox", (0, 0, 0, 0)))
        if not MIN_INSTANCE_AREA <= area_fraction <= MAX_INSTANCE_AREA:
            continue
        if min(width, height) < MIN_PARENT_SIDE / 2:
            continue
        candidates.append(annotation)
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    regions = []
    image_paths = {}
    for annotation in candidates:
        record = images[int(annotation["image_id"])]
        mask = _decode_coco_segmentation(
            annotation, int(record["height"]), int(record["width"]), mask_utils
        )
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        object_width, object_height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        pad_x = round(PARENT_PADDING * object_width)
        pad_y = round(PARENT_PADDING * object_height)
        box = (
            max(0, int(xs.min()) - pad_x),
            max(0, int(ys.min()) - pad_y),
            min(int(record["width"]), int(xs.max()) + 1 + pad_x),
            min(int(record["height"]), int(ys.max()) + 1 + pad_y),
        )
        if min(box[2] - box[0], box[3] - box[1]) < MIN_PARENT_SIDE:
            continue
        image_id = int(annotation["image_id"])
        if image_id not in image_paths:
            image_paths[image_id] = _resolve_coco_image(coco, split, record["file_name"])
        category_id = int(annotation["category_id"])
        regions.append(CocoRegion(
            image_id=image_id,
            annotation_id=int(annotation["id"]),
            category_id=category_id,
            category=categories[category_id],
            image_path=str(image_paths[image_id]),
            box=box,
        ))
        if len(regions) == num_objects:
            break
    if len(regions) < num_objects:
        raise ValueError(
            f"Only {len(regions)} COCO objects passed the filters; requested {num_objects}"
        )
    return regions


def load_teacher(path):
    """Load the frozen teacher backbone and patch-prediction head."""
    backbone, metadata = load_backbone(path, "teacher", "auto")
    checkpoint = _torch_load(path)
    head, prototypes = _build_teacher_head(
        backbone, checkpoint, _teacher_head_state(_teacher_state(checkpoint))
    )
    backbone = backbone.to(DEVICE).eval().requires_grad_(False)
    head = head.to(DEVICE).eval().requires_grad_(False)
    return backbone, head, metadata, prototypes


def _grid_shape(parts):
    if parts == 2:
        return 1, 2
    side = math.isqrt(parts)
    if side * side != parts:
        raise ValueError("Part counts must be 2 or perfect squares")
    return side, side


def partition_box(box, parts):
    """Divide a rectangle into exhaustive, non-overlapping grid cells."""
    rows, columns = _grid_shape(parts)
    left, top, right, bottom = box
    xs = np.rint(np.linspace(left, right, columns + 1)).astype(int)
    ys = np.rint(np.linspace(top, bottom, rows + 1)).astype(int)
    boxes = []
    for row in range(rows):
        for column in range(columns):
            child = (
                int(xs[column]), int(ys[row]),
                int(xs[column + 1]), int(ys[row + 1]),
            )
            if child[2] <= child[0] or child[3] <= child[1]:
                raise ValueError(f"Parent {box} is too small for K={parts}")
            boxes.append(child)
    return boxes


def _prepare_crop(image, box):
    crop = image.crop(box).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
    tensor = TF.to_tensor(crop)
    return TF.normalize(tensor, (0.485, .456, .406), (.229, .224, .225))


@torch.inference_mode()
def encode_region(image, box, backbone, head, patch_size):
    """Encode one crop independently and average its patch distributions."""
    tensor = _prepare_crop(image, box)[None].to(DEVICE)
    tokens = backbone(tensor, return_all_tokens=True)
    spatial = tokens[:, _num_special_tokens(backbone):]
    _, logits = head(torch.cat((tokens[:, :1], spatial), dim=1))
    expected_patches = (INPUT_SIZE // patch_size) ** 2
    if logits.ndim != 3 or logits.shape[:2] != (1, expected_patches):
        raise ValueError(
            f"Patch head returned {tuple(logits.shape)}; expected "
            f"[1, {expected_patches}, prototypes]"
        )
    probabilities = (logits[0].float() / TEMPERATURE).softmax(dim=-1)
    distribution = probabilities.mean(dim=0).double().cpu().numpy()
    return distribution / distribution.sum()


def jensen_shannon(p, q):
    """Natural-log Jensen--Shannon divergence, bounded by ln(2)."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("Jensen--Shannon inputs must be same-shaped vectors")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p < 0) or np.any(q < 0):
        raise ValueError("Jensen--Shannon inputs must be finite and nonnegative")
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("Jensen--Shannon inputs must have positive mass")
    p, q = p / p.sum(), q / q.sum()
    midpoint = .5 * (p + q)

    def kl(distribution):
        positive = distribution > 0
        return float(np.sum(
            distribution[positive] * np.log(distribution[positive] / midpoint[positive])
        ))

    return .5 * (kl(p) + kl(q))


def evaluate_region(region, model_name, backbone, head, patch_size):
    """Measure reconstruction error for every configured decomposition."""
    with Image.open(region.image_path) as source:
        image = source.convert("RGB")
    parent_distribution = encode_region(image, region.box, backbone, head, patch_size)
    rows = []
    for parts in PART_COUNTS:
        child_distributions = np.stack([
            encode_region(image, child_box, backbone, head, patch_size)
            for child_box in partition_box(region.box, parts)
        ])
        reconstruction = child_distributions.mean(axis=0)
        rows.append(Measurement(
            model=model_name,
            image_id=region.image_id,
            annotation_id=region.annotation_id,
            category_id=region.category_id,
            parts=parts,
            composition_js=jensen_shannon(parent_distribution, reconstruction),
        ))
    return rows


def bootstrap_summary(
    measurements,
    samples=BOOTSTRAP_SAMPLES,
    confidence=CONFIDENCE,
    seed=SEED,
):
    """Cluster-bootstrap COCO images while retaining all sampled objects."""
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Bootstrap samples must be positive and confidence in (0, 1)")
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    models = list(dict.fromkeys(row.model for row in measurements))
    summary = []
    for model in models:
        for parts in PART_COUNTS:
            subset = [
                row for row in measurements
                if row.model == model and row.parts == parts
            ]
            by_image = {}
            for row in subset:
                by_image.setdefault(row.image_id, []).append(row.composition_js)
            image_ids = sorted(by_image)
            if len(image_ids) < 2:
                raise ValueError("Bootstrap intervals require at least two COCO images")
            draws = np.empty(samples, dtype=np.float64)
            for index in range(samples):
                sampled_ids = rng.choice(image_ids, size=len(image_ids), replace=True)
                draws[index] = np.mean([
                    value for image_id in sampled_ids for value in by_image[int(image_id)]
                ])
            low, high = np.quantile(draws, (alpha, 1 - alpha))
            summary.append({
                "model": model,
                "parts": parts,
                "mean_js": float(np.mean([row.composition_js for row in subset])),
                "ci_low": float(low),
                "ci_high": float(high),
                "objects": len(subset),
                "images": len(image_ids),
            })
    return summary


def render_plot(summary, path):
    colors = ("#64748B", "#2563EB")
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    models = list(dict.fromkeys(row["model"] for row in summary))
    for model, color in zip(models, colors):
        rows = sorted(
            (row for row in summary if row["model"] == model),
            key=lambda row: row["parts"],
        )
        x = np.array([row["parts"] for row in rows])
        mean = np.array([row["mean_js"] for row in rows])
        low = np.array([row["ci_low"] for row in rows])
        high = np.array([row["ci_high"] for row in rows])
        axis.plot(x, mean, marker="o", linewidth=2.2, color=color, label=model)
        axis.fill_between(x, low, high, color=color, alpha=.17, linewidth=0)
    axis.set(
        xlabel="Number of independently encoded parts K",
        ylabel="Mean JS composition error ↓",
        xticks=PART_COUNTS,
        title="Independent part-to-whole re-encoding",
    )
    axis.grid(axis="y", alpha=.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_configuration():
    if len(CHECKPOINTS) != 2:
        raise ValueError("Configure exactly two checkpoints: matched control and region-trained")
    if tuple(PART_COUNTS) != (2, 4, 9):
        raise ValueError("This protocol evaluates exactly K=(2, 4, 9)")
    if INPUT_SIZE < 1 or not math.isfinite(TEMPERATURE) or TEMPERATURE <= 0:
        raise ValueError("INPUT_SIZE and TEMPERATURE must be positive")
    missing = [
        str(path) for path in CHECKPOINTS.values()
        if not Path(path).expanduser().is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")


def main():
    validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_coco_regions()
    measurements = []
    model_metadata = {}
    for model_name, checkpoint in CHECKPOINTS.items():
        print(f"Loading {model_name}: {checkpoint}", flush=True)
        backbone, head, metadata, prototypes = load_teacher(checkpoint)
        patch_size = int(metadata["patch_size"])
        if INPUT_SIZE % patch_size:
            raise ValueError(
                f"INPUT_SIZE={INPUT_SIZE} is not divisible by patch size {patch_size}"
            )
        checkpoint_state = _torch_load(checkpoint)
        model_metadata[model_name] = {
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "patch_size": patch_size,
            "prototypes": int(prototypes),
            "checkpoint_region_normalization": _checkpoint_argument(
                checkpoint_state, "region_normalization", None
            ),
        }
        for index, region in enumerate(regions, 1):
            measurements.extend(
                evaluate_region(region, model_name, backbone, head, patch_size)
            )
            if index % 25 == 0 or index == len(regions):
                print(f"  {index}/{len(regions)} objects", flush=True)
        del backbone, head
        if str(DEVICE).startswith("cuda"):
            torch.cuda.empty_cache()

    summary = bootstrap_summary(measurements)
    _write_csv(OUTPUT_DIR / "measurements.csv", [asdict(row) for row in measurements])
    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "regions.csv", [{
        "image_id": region.image_id,
        "annotation_id": region.annotation_id,
        "category_id": region.category_id,
        "category": region.category,
        "image_path": region.image_path,
        "parent_box_left": region.box[0],
        "parent_box_top": region.box[1],
        "parent_box_right": region.box[2],
        "parent_box_bottom": region.box[3],
    } for region in regions])
    render_plot(summary, OUTPUT_DIR / "composition_reencoding.png")
    protocol = {
        "experiment": "independent part-to-whole compositional re-encoding",
        "models": model_metadata,
        "coco_split": COCO_SPLIT,
        "objects": len(regions),
        "images": len({region.image_id for region in regions}),
        "part_counts": list(PART_COUNTS),
        "input_size": INPUT_SIZE,
        "softmax_temperature": TEMPERATURE,
        "parent_padding": PARENT_PADDING,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "confidence": CONFIDENCE,
        "bootstrap_unit": "COCO image (all sampled objects retained within a cluster)",
        "seed": SEED,
        "js_log_base": "natural",
        "parent_definition": "padded bounding rectangle of a valid COCO instance mask",
        "reconstruction": "unweighted mean of independently encoded child distributions",
        "independence": "one crop per forward call",
        "scale_stress_test": "each child is independently resized to model input resolution",
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Wrote results to {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
