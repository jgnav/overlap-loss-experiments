#!/usr/bin/env python3
"""Independent part-to-whole re-encoding experiment for iBOT checkpoints.

Edit the configuration below and run this file directly. The same held-out COCO
regions are used for every model. Every parent, child, and shuffled child crop
is encoded in its own forward pass.
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
    "iBOT initialization": Path("checkpoints/checkpoint_source1000.pth"),
    "iBOT + 200 epoch control": Path("checkpoints/checkpoint_control0200.pth"),
    "Ours + 200 epochs": Path("checkpoints/checkpoint_source1000_continuation0200.pth"),
}
DATASETS_ROOT = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/datasets")
COCO_SPLIT = "val2017"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "composition_reencoding"
DEVICE = "cuda"
INPUT_SIZE = 224
PART_COUNTS = (2, 4, 9)
NUM_IMAGES = 250
PARENTS_PER_IMAGE = 1
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
    mask: np.ndarray
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class RegionPair:
    target: CocoRegion
    donor: CocoRegion


@dataclass(frozen=True)
class Measurement:
    model: str
    image_id: int
    annotation_id: int
    donor_image_id: int
    parts: int
    composition_js: float
    shuffled_js: float
    specificity_gap: float


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
        raise FileNotFoundError(f"Expected exactly one COCO image for {filename}; found {matches}")
    return matches[0]


def load_coco_regions(
    datasets_root=DATASETS_ROOT, split=COCO_SPLIT, num_images=NUM_IMAGES,
    parents_per_image=PARENTS_PER_IMAGE, seed=SEED,
):
    """Load deterministic held-out COCO instance masks and padded parent boxes."""
    if split not in {"train2017", "val2017"}:
        raise ValueError("COCO_SPLIT must be 'train2017' or 'val2017'")
    if num_images < 2 or parents_per_image < 1:
        raise ValueError("Need at least two images and one parent per image")
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
    by_image: dict[int, list[dict]] = {}
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        record = images.get(image_id)
        if record is None or annotation.get("iscrowd", 0) or not annotation.get("segmentation"):
            continue
        area_fraction = float(annotation.get("area", 0)) / float(record["height"] * record["width"])
        _, _, width, height = map(float, annotation.get("bbox", (0, 0, 0, 0)))
        if not MIN_INSTANCE_AREA <= area_fraction <= MAX_INSTANCE_AREA:
            continue
        if min(width, height) < MIN_PARENT_SIDE / 2:
            continue
        by_image.setdefault(image_id, []).append(annotation)

    rng = np.random.default_rng(seed)
    image_ids = np.array(sorted(by_image))
    rng.shuffle(image_ids)
    regions, used_images = [], 0
    for image_id in image_ids:
        record = images[int(image_id)]
        image_path = _resolve_coco_image(coco, split, record["file_name"])
        candidates = sorted(
            by_image[int(image_id)],
            key=lambda row: (-float(row.get("area", 0)), int(row["id"])),
        )[:parents_per_image]
        image_regions = []
        for annotation in candidates:
            mask = _decode_coco_segmentation(
                annotation, int(record["height"]), int(record["width"]), mask_utils
            )
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            object_width = xs.max() - xs.min() + 1
            object_height = ys.max() - ys.min() + 1
            pad_x, pad_y = round(PARENT_PADDING * object_width), round(PARENT_PADDING * object_height)
            box = (
                max(0, int(xs.min()) - pad_x), max(0, int(ys.min()) - pad_y),
                min(int(record["width"]), int(xs.max()) + 1 + pad_x),
                min(int(record["height"]), int(ys.max()) + 1 + pad_y),
            )
            if min(box[2] - box[0], box[3] - box[1]) < MIN_PARENT_SIDE:
                continue
            image_regions.append(CocoRegion(
                int(image_id), int(annotation["id"]), int(annotation["category_id"]),
                categories[int(annotation["category_id"])], str(image_path), mask, box,
            ))
        if image_regions:
            regions.extend(image_regions)
            used_images += 1
        if used_images == num_images:
            break
    if used_images < num_images:
        raise ValueError(f"Only {used_images} COCO images passed the region filters; requested {num_images}")
    return regions


def pair_shuffled_regions(regions, seed=SEED):
    """Assign each target a deterministic donor from a different image."""
    if len({region.image_id for region in regions}) < 2:
        raise ValueError("Shuffled controls require at least two source images")
    rng = np.random.default_rng(seed + 1)
    donors = list(regions)
    for _ in range(10_000):
        rng.shuffle(donors)
        if all(a.image_id != b.image_id for a, b in zip(regions, donors)):
            return [RegionPair(a, b) for a, b in zip(regions, donors)]
    donors = sorted(regions, key=lambda item: (item.image_id, item.annotation_id))
    for offset in range(1, len(donors)):
        shifted = donors[offset:] + donors[:offset]
        if all(a.image_id != b.image_id for a, b in zip(regions, shifted)):
            return [RegionPair(a, b) for a, b in zip(regions, shifted)]
    raise RuntimeError("Could not construct a cross-image donor assignment")


def load_teacher(path):
    """Load the frozen teacher backbone and patch head from one checkpoint."""
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
        raise ValueError("PART_COUNTS entries must be 2 or perfect squares")
    return side, side


def partition_box(box, parts):
    """Tile a parent rectangle without overlap or dropped source pixels."""
    rows, columns = _grid_shape(parts)
    left, top, right, bottom = box
    xs = np.rint(np.linspace(left, right, columns + 1)).astype(int)
    ys = np.rint(np.linspace(top, bottom, rows + 1)).astype(int)
    boxes, areas = [], []
    for row in range(rows):
        for column in range(columns):
            child = (int(xs[column]), int(ys[row]), int(xs[column + 1]), int(ys[row + 1]))
            area = (child[2] - child[0]) * (child[3] - child[1])
            if area <= 0:
                raise ValueError(f"Parent {box} is too small for K={parts}")
            boxes.append(child)
            areas.append(area)
    weights = np.asarray(areas, dtype=np.float64)
    weights /= weights.sum()
    return boxes, weights


def _prepare_crop(image, box):
    crop = image.crop(box).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
    tensor = TF.to_tensor(crop)
    return TF.normalize(tensor, (0.485, .456, .406), (.229, .224, .225))


@torch.inference_mode()
def encode_region(image, box, backbone, head, patch_size):
    """Encode one crop in one forward call and average its patch softmaxes."""
    tensor = _prepare_crop(image, box)[None].to(DEVICE)
    tokens = backbone(tensor, return_all_tokens=True)
    spatial = tokens[:, _num_special_tokens(backbone):]
    _, logits = head(torch.cat((tokens[:, :1], spatial), dim=1))
    expected = (INPUT_SIZE // patch_size) ** 2
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != expected:
        raise ValueError(f"Patch head returned {tuple(logits.shape)}; expected [1, {expected}, prototypes]")
    distribution = (logits[0].float() / TEMPERATURE).softmax(dim=-1).mean(dim=0)
    distribution = distribution.double().cpu().numpy()
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
    def kl(x):
        positive = x > 0
        return float(np.sum(x[positive] * np.log(x[positive] / midpoint[positive])))
    return .5 * (kl(p) + kl(q))


def evaluate_pair(pair, model_name, backbone, head, patch_size):
    """Evaluate one target parent against own and donor child crops."""
    with Image.open(pair.target.image_path) as source:
        target_image = source.convert("RGB")
    with Image.open(pair.donor.image_path) as source:
        donor_image = source.convert("RGB")
    parent_q = encode_region(target_image, pair.target.box, backbone, head, patch_size)
    rows = []
    for parts in PART_COUNTS:
        child_boxes, weights = partition_box(pair.target.box, parts)
        donor_boxes, _ = partition_box(pair.donor.box, parts)
        child_q = np.stack([encode_region(target_image, box, backbone, head, patch_size) for box in child_boxes])
        shuffled_q = np.stack([encode_region(donor_image, box, backbone, head, patch_size) for box in donor_boxes])
        composition_js = jensen_shannon(parent_q, weights @ child_q)
        shuffled_js = jensen_shannon(parent_q, weights @ shuffled_q)
        rows.append(Measurement(
            model_name, pair.target.image_id, pair.target.annotation_id,
            pair.donor.image_id, parts, composition_js, shuffled_js,
            shuffled_js - composition_js,
        ))
    return rows


def bootstrap_summary(measurements, samples=BOOTSTRAP_SAMPLES, confidence=CONFIDENCE, seed=SEED):
    """Cluster bootstrap over images, averaging parents within each image."""
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Bootstrap samples must be positive and confidence in (0, 1)")
    rng = np.random.default_rng(seed)
    result = []
    models = list(dict.fromkeys(row.model for row in measurements))
    alpha = (1 - confidence) / 2
    metrics = ("composition_js", "shuffled_js", "specificity_gap")
    for model in models:
        for parts in PART_COUNTS:
            subset = [row for row in measurements if row.model == model and row.parts == parts]
            image_ids = sorted({row.image_id for row in subset})
            if len(image_ids) < 2:
                raise ValueError("Bootstrap intervals require at least two evaluated images")
            image_values = np.array([
                [np.mean([getattr(row, metric) for row in subset if row.image_id == image_id]) for metric in metrics]
                for image_id in image_ids
            ])
            draws = image_values[rng.integers(0, len(image_values), (samples, len(image_values)))].mean(1)
            means = image_values.mean(0)
            lower, upper = np.quantile(draws, (alpha, 1 - alpha), axis=0)
            for index, metric in enumerate(metrics):
                result.append({
                    "model": model, "parts": parts, "metric": metric,
                    "mean": float(means[index]), "ci_low": float(lower[index]),
                    "ci_high": float(upper[index]), "images": len(image_ids),
                })
    return result


def render_plot(summary, path):
    colors = ("#64748B", "#D97706", "#2563EB", "#059669", "#DC2626")
    models = list(dict.fromkeys(row["model"] for row in summary))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for model, color in zip(models, colors):
        for axis, metric, title, ylabel in (
            (axes[0], "composition_js", "Part-to-whole re-encoding", "JS composition error ↓"),
            (axes[1], "specificity_gap", "Shuffled-parts control", "Composition specificity gap ↑"),
        ):
            rows = sorted(
                (row for row in summary if row["model"] == model and row["metric"] == metric),
                key=lambda row: row["parts"],
            )
            x = np.array([row["parts"] for row in rows])
            y = np.array([row["mean"] for row in rows])
            low = np.array([row["ci_low"] for row in rows])
            high = np.array([row["ci_high"] for row in rows])
            axis.plot(x, y, marker="o", linewidth=2, color=color, label=model)
            axis.fill_between(x, low, high, color=color, alpha=.16, linewidth=0)
            axis.set(title=title, xlabel="Number of independently encoded parts K", ylabel=ylabel)
            axis.set_xticks(PART_COUNTS)
            axis.grid(axis="y", alpha=.22)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"COCO {COCO_SPLIT} · image-level {CONFIDENCE:.0%} bootstrap intervals", fontsize=12)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_configuration():
    if tuple(PART_COUNTS) != (2, 4, 9):
        raise ValueError("This protocol evaluates exactly K=(2, 4, 9)")
    if INPUT_SIZE < 1 or not math.isfinite(TEMPERATURE) or TEMPERATURE <= 0:
        raise ValueError("INPUT_SIZE and TEMPERATURE must be positive")
    if not CHECKPOINTS:
        raise ValueError("Configure at least one checkpoint")
    missing = [str(path) for path in CHECKPOINTS.values() if not Path(path).expanduser().is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")


def main():
    validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_coco_regions()
    pairs = pair_shuffled_regions(regions)
    measurements, model_metadata = [], {}
    for model_name, checkpoint in CHECKPOINTS.items():
        print(f"Loading {model_name}: {checkpoint}", flush=True)
        backbone, head, metadata, prototypes = load_teacher(checkpoint)
        patch_size = int(metadata["patch_size"])
        if INPUT_SIZE % patch_size:
            raise ValueError(f"INPUT_SIZE={INPUT_SIZE} is not divisible by patch size {patch_size}")
        checkpoint_state = _torch_load(checkpoint)
        model_metadata[model_name] = {
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "patch_size": patch_size, "prototypes": int(prototypes),
            "checkpoint_region_normalization": _checkpoint_argument(checkpoint_state, "region_normalization", None),
        }
        for index, pair in enumerate(pairs, 1):
            measurements.extend(evaluate_pair(pair, model_name, backbone, head, patch_size))
            if index % 25 == 0 or index == len(pairs):
                print(f"  {index}/{len(pairs)} parents", flush=True)
        del backbone, head
        if str(DEVICE).startswith("cuda"):
            torch.cuda.empty_cache()

    summary = bootstrap_summary(measurements)
    _write_csv(OUTPUT_DIR / "measurements.csv", [asdict(row) for row in measurements])
    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "regions.csv", [{
        "image_id": pair.target.image_id,
        "annotation_id": pair.target.annotation_id,
        "category_id": pair.target.category_id,
        "category": pair.target.category,
        "image_path": pair.target.image_path,
        "parent_box_left": pair.target.box[0],
        "parent_box_top": pair.target.box[1],
        "parent_box_right": pair.target.box[2],
        "parent_box_bottom": pair.target.box[3],
        "donor_image_id": pair.donor.image_id,
        "donor_annotation_id": pair.donor.annotation_id,
    } for pair in pairs])
    render_plot(summary, OUTPUT_DIR / "composition_reencoding.png")
    protocol = {
        "experiment": "independent part-to-whole compositional re-encoding",
        "models": model_metadata, "coco_split": COCO_SPLIT,
        "images": len({row.image_id for row in measurements}), "parents": len(regions),
        "part_counts": list(PART_COUNTS), "input_size": INPUT_SIZE,
        "softmax_temperature": TEMPERATURE, "parent_padding": PARENT_PADDING,
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "confidence": CONFIDENCE,
        "seed": SEED, "js_log_base": "natural",
        "parent_definition": "padded bounding rectangle of a valid COCO instance mask",
        "independence": "one crop per forward call",
        "shuffled_control": "same partition index from a different COCO image",
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Wrote results to {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
