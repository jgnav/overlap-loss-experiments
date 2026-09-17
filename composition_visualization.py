#!/usr/bin/env python3
"""Paper figure of A+B patch-distribution composition, without projection.

Usage: python composition_visualization.py
Edit the configuration below. Empty REFERENCE_A/B lists select independent
reference instances automatically from COCO.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

from evaluation.utils.common import load_backbone
from losses.sinkhorn import sinkhorn_log_probabilities
from patch_concept_visualization import (
    _build_teacher_head, _checkpoint_argument, _num_special_tokens,
    _teacher_head_state, _teacher_state, _torch_load,
)


# ---- Hard-coded experiment configuration -----------------------------------
# All configuration lives here; no command-line arguments are required.
CHECKPOINT = Path("checkpoints/checkpoint_source1000_continuation0200.pth")
COCO_IMAGE_ID = 240684
DATASETS_ROOT = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/datasets")
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "composition"
DEVICE = "cuda"  # Set to "cuda" on your GPU machine if desired.
CONCEPT_A = "A"  # For example, "Dog"
CONCEPT_B = "B"  # For example, "Car"
BACKGROUND_COLOR = (0, 0, 0)
OBJECT_A_COLOR = (255, 0, 0)  # Colors used in the generated COCO mask.
OBJECT_B_COLOR = (0, 255, 0)
COCO_ANNOTATION_A = None  # Optional exact instance annotation ID.
COCO_ANNOTATION_B = None  # Otherwise match CONCEPT_A/B category names, or auto-select.
LONG_SIDE = 560
PATCH_PURITY = 0.9  # Fraction of a patch occupied by one object; must exceed .5.
MIXED_CROP_PADDING = 0.10  # Fractional padding around the union of both objects.
AUTO_REFERENCE_COUNT = 4  # Per concept when REFERENCE_A/B are left empty.
HEATMAP_PROTOTYPES = 32  # Illustration only; all dimensions are used in fits.
CONCENTRATION_K = (8, 32, 128, 512)
MIXTURE_PATCHES = 64  # Common count, reduced to available patches; multiple of 4.
SEED = 0
DPI = 300
NORMALIZATION_OVERRIDE = None  # None uses checkpoint region_normalization.
TEMPERATURE_OVERRIDE = None  # None uses checkpoint region_temp (fallback .1).
COLORS = ("#2676B8", "#D97924", "#A5ADB6")  # A contribution / B contribution / residual


@dataclass(frozen=True)
class ReferenceRegion:
    image: str | Path
    mask: str | Path
    color: tuple[int, int, int]


# Leave both lists empty to select AUTO_REFERENCE_COUNT independent images per
# concept from COCO. Explicit entries override automatic selection. Each entry
# defines ONE pure object region; masks may contain additional unrelated colors.
# Multiple regions are averaged equally, irrespective of their patch counts.
# At least two regions per concept are required so pure controls can use
# leave-one-reference-out fingerprints instead of fitting a sample to itself.
REFERENCE_A: list[ReferenceRegion] = [
    # ReferenceRegion("/path/dog_1.jpg", "/path/dog_1_mask.png", (255, 0, 0)),
]
REFERENCE_B: list[ReferenceRegion] = [
    # ReferenceRegion("/path/car_1.jpg", "/path/car_1_mask.png", (0, 255, 0)),
]
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    image: Image.Image
    mask: np.ndarray
    grid: tuple[int, int]
    logits: torch.Tensor  # [patches, prototypes], raw teacher patch-head logits
    geometry: dict


@dataclass
class Composition:
    mu_a: np.ndarray
    mu_b: np.ndarray
    mixed: np.ndarray
    reference_means_a: np.ndarray
    reference_means_b: np.ndarray
    heatmap_indices: np.ndarray
    heatmap_signs: np.ndarray
    concentration: dict
    fit_shares: np.ndarray  # leave-one-out A, leave-one-out B, observed A+B
    fit_coefficients: np.ndarray
    fit_residual_l1: np.ndarray
    fit_cosine: np.ndarray
    mixed_reconstruction: np.ndarray
    mixed_residual: np.ndarray
    mixture_means: np.ndarray
    mixture_fit_shares: np.ndarray
    mixture_counts: list[tuple[int, int]]
    mixture_indices: list[np.ndarray]
    selected_a: np.ndarray
    selected_b: np.ndarray
    counts: tuple[int, int]
    identity_error: float


def _decode_coco_segmentation(annotation, height, width, mask_utils):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    result = mask_utils.decode(rle).astype(bool)
    if result.shape != (height, width):
        raise ValueError("Decoded COCO mask has incorrect dimensions")
    return result


def find_coco_pair(image_id, datasets_root):
    """Look only in the configured COCO tree and rasterize two object masks."""
    root = Path(datasets_root).expanduser().resolve()
    if image_id < 0 or not root.is_dir():
        raise ValueError("Require a nonnegative COCO image ID and an existing datasets root")
    coco_root = root if root.name.casefold() == "coco" else root / "coco"
    if not coco_root.is_dir():
        raise FileNotFoundError(
            f"COCO directory not found at {coco_root}; DATASETS_ROOT may be the "
            "datasets directory or the COCO directory itself"
        )
    matches = []
    for split in ("train2017", "val2017"):
        path = coco_root / "annotations" / f"instances_{split}.json"
        if not path.is_file():
            continue
        with path.open() as stream:
            data = json.load(stream)
        records = [r for r in data.get("images", []) if r["id"] == image_id]
        if records:
            matches.append((path, split, records[0], data))
    if not matches:
        raise FileNotFoundError(
            f"COCO image {image_id} not found in train2017/val2017 instance "
            f"annotations at {coco_root / 'annotations'}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Image {image_id} occurs in multiple annotation files: "
            f"{[str(x[0]) for x in matches]}"
        )
    annotation_path, split, record, data = matches[0]
    filename = Path(record["file_name"]).name
    image_candidates = (
        coco_root / "images" / split / filename,  # prepare_data.py layout
        coco_root / split / filename,             # official archive layout
    )
    images = [path for path in image_candidates if path.is_file()]
    if not images:
        raise FileNotFoundError(
            f"COCO image file {filename} is missing; expected "
            f"{image_candidates[0]} or {image_candidates[1]}"
        )
    if len(images) > 1:
        raise ValueError(f"COCO image exists in both supported layouts: {images}")
    categories = {c["id"]: c["name"] for c in data["categories"]}
    candidates = [a for a in data["annotations"] if a["image_id"] == image_id
                  and not a.get("iscrowd", 0) and a.get("segmentation")]
    candidates.sort(key=lambda a: (-a.get("area", 0), a["id"]))
    def choose(annotation_id, concept, excluded=None):
        options = [a for a in candidates if excluded is None or a["category_id"] != excluded["category_id"]]
        if annotation_id is not None:
            options = [a for a in options if a["id"] == annotation_id]
        elif concept not in ("A", "B"):
            options = [a for a in options if categories[a["category_id"]].casefold() == concept.casefold()]
        if not options:
            raise ValueError(f"No eligible COCO instance for {concept} (annotation ID {annotation_id}); available: {[(a['id'], categories[a['category_id']]) for a in candidates]}")
        return options[0]
    a = choose(COCO_ANNOTATION_A, CONCEPT_A)
    b = choose(COCO_ANNOTATION_B, CONCEPT_B, a)
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("COCO polygon/RLE decoding requires pycocotools; install it in the active Python environment") from exc
    height, width = int(record["height"]), int(record["width"])
    with Image.open(images[0]) as image:
        if image.size != (width, height):
            raise ValueError("COCO annotation dimensions differ from the local image")
    ma = _decode_coco_segmentation(a, height, width, mask_utils)
    mb = _decode_coco_segmentation(b, height, width, mask_utils)
    ambiguous = ma & mb
    ma, mb = ma & ~ambiguous, mb & ~ambiguous
    if not ma.any() or not mb.any():
        raise ValueError("Selected COCO objects have no exclusive mask pixels")
    color_a, color_b = OBJECT_A_COLOR, OBJECT_B_COLOR
    if color_a is None or color_b is None or color_a == color_b or BACKGROUND_COLOR in (color_a, color_b):
        raise ValueError("Configure distinct OBJECT_A_COLOR, OBJECT_B_COLOR and BACKGROUND_COLOR RGB values")
    mask = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)
    mask[ma], mask[mb] = color_a, color_b
    path = Path(OUTPUT_DIR) / f"coco_{image_id:012d}_{a['id']}_{b['id']}_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)
    provenance = {"image_id": image_id, "datasets_root": str(root),
                  "coco_root": str(coco_root), "split": split,
                  "annotations": str(annotation_path),
                  "objects": [{"annotation_id": obj["id"], "category_id": obj["category_id"],
                               "category": categories[obj["category_id"]]} for obj in (a, b)],
                  "ambiguous_pixels_excluded": int(ambiguous.sum())}
    print(f"COCO {image_id}: A={provenance['objects'][0]}, B={provenance['objects'][1]}", flush=True)
    return images[0], path, provenance


def find_coco_references(coco, count=AUTO_REFERENCE_COUNT):
    """Create deterministic independent object references from COCO only."""
    if count < 2:
        raise ValueError("AUTO_REFERENCE_COUNT must be at least two")
    coco_root = Path(coco["coco_root"])
    target_image_id = int(coco["image_id"])
    category_ids = [int(obj["category_id"]) for obj in coco["objects"]]
    colors = (OBJECT_A_COLOR, OBJECT_B_COLOR)
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("Automatic COCO references require pycocotools") from exc
    candidates = {category_id: [] for category_id in category_ids}
    for split in ("train2017", "val2017"):
        annotation_path = coco_root / "annotations" / f"instances_{split}.json"
        if not annotation_path.is_file():
            continue
        with annotation_path.open() as stream:
            data = json.load(stream)
        images = {int(item["id"]): item for item in data.get("images", [])}
        for annotation in data.get("annotations", []):
            category_id = int(annotation["category_id"])
            image_id = int(annotation["image_id"])
            if (category_id in candidates and image_id != target_image_id
                    and not annotation.get("iscrowd", 0)
                    and annotation.get("segmentation") and image_id in images):
                candidates[category_id].append((
                    -float(annotation.get("area", 0)), int(annotation["id"]),
                    split, annotation, images[image_id], annotation_path,
                ))
    output = []
    reference_dir = Path(OUTPUT_DIR) / "coco_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for category_id, color in zip(category_ids, colors):
        references, used_images = [], set()
        for _, _, split, annotation, record, annotation_path in sorted(candidates[category_id]):
            image_id = int(record["id"])
            if image_id in used_images:
                continue
            filename = Path(record["file_name"]).name
            paths = (coco_root / "images" / split / filename,
                     coco_root / split / filename)
            image_paths = [path for path in paths if path.is_file()]
            if len(image_paths) != 1:
                continue
            height, width = int(record["height"]), int(record["width"])
            selected = _decode_coco_segmentation(
                annotation, height, width, mask_utils
            )
            if not selected.any():
                continue
            mask = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)
            mask[selected] = color
            mask_path = reference_dir / (
                f"coco_{image_id:012d}_{annotation['id']}_mask.png"
            )
            Image.fromarray(mask).save(mask_path)
            references.append(ReferenceRegion(image_paths[0], mask_path, color))
            used_images.add(image_id)
            if len(references) == count:
                break
        if len(references) < count:
            raise ValueError(
                f"COCO contains only {len(references)} usable independent images "
                f"for category ID {category_id}; need {count}"
            )
        output.append(references)
    return tuple(output)


def image_digest(path):
    """Detect byte-identical decoded images even if copied/re-encoded losslessly."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def validate_inputs(checkpoint, image, mask, references_a=None, references_b=None):
    references_a = REFERENCE_A if references_a is None else references_a
    references_b = REFERENCE_B if references_b is None else references_b
    if len(references_a) < 2 or len(references_b) < 2:
        raise ValueError(
            "Configure at least two independent REFERENCE_A and REFERENCE_B "
            "regions for fingerprints and leave-one-out controls"
        )
    if (not .5 < PATCH_PURITY <= 1 or HEATMAP_PROTOTYPES < 1
            or MIXTURE_PATCHES < 4 or LONG_SIDE < 1
            or not 0 <= MIXED_CROP_PADDING <= 1):
        raise ValueError(
            "Require .5 < PATCH_PURITY <= 1, HEATMAP_PROTOTYPES >= 1, "
            "MIXTURE_PATCHES >= 4, LONG_SIDE >= 1, and crop padding in [0, 1]"
        )
    if not CONCENTRATION_K or any(type(k) is not int or k <= 0 for k in CONCENTRATION_K):
        raise ValueError("CONCENTRATION_K must contain positive integers")
    paths = [checkpoint, image, mask]
    for reference in (*references_a, *references_b):
        paths.extend((reference.image, reference.mask))
        if len(reference.color) != 3 or any(not 0 <= value <= 255 for value in reference.color):
            raise ValueError("Reference colors must be RGB triples in [0, 255]")
    for path in paths:
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(path)
    target_digest = image_digest(image)
    for reference in (*references_a, *references_b):
        if image_digest(Path(reference.image).expanduser()) == target_digest:
            raise ValueError("Calibration leakage: a reference image matches the displayed image")


def resolve_colors(mask):
    if (OBJECT_A_COLOR is None) != (OBJECT_B_COLOR is None):
        raise ValueError("Set both OBJECT_A_COLOR and OBJECT_B_COLOR, or neither")
    if OBJECT_A_COLOR is None:
        colors = [tuple(map(int, row)) for row in np.unique(mask.reshape(-1, 3), axis=0)
                  if tuple(row) != BACKGROUND_COLOR]
        if len(colors) != 2:
            raise ValueError(f"Expected two non-background colors, found {len(colors)}; set explicit OBJECT_A/B_COLOR for this mask")
        a, b = colors
    else:
        a, b = tuple(OBJECT_A_COLOR), tuple(OBJECT_B_COLOR)
    if a == b or a == BACKGROUND_COLOR or b == BACKGROUND_COLOR:
        raise ValueError("A, B and background colors must differ")
    for color in (a, b):
        if not np.all(mask == color, axis=-1).any():
            raise ValueError(f"Object mask color {color} is absent")
    return a, b


def _prepare_pair_images(image, mask, patch_size, long_side=None):
    """Resize two PIL images together, then pad to a complete patch grid."""
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    if image.size != mask.size:
        raise ValueError(f"Image and mask dimensions differ: {image.size} vs {mask.size}")
    long_side = LONG_SIDE if long_side is None else long_side
    original_size = image.size
    scale = long_side / max(image.size)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    image = image.resize(size, Image.Resampling.BICUBIC)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    padded_size = tuple(math.ceil(side / patch_size) * patch_size for side in size)
    canvas = Image.new("RGB", padded_size, (124, 116, 104))
    canvas.paste(image, (0, 0))
    mask_canvas = Image.new("RGB", padded_size, BACKGROUND_COLOR)
    mask_canvas.paste(mask, (0, 0))
    geometry = {"original_size": original_size, "resized_size": size,
                "padded_size": padded_size, "padding": "right and bottom"}
    return canvas, np.asarray(mask_canvas), geometry


def prepare_pair(image_path, mask_path, patch_size, long_side=None):
    """Load an aligned image/mask pair and preserve all content during resize."""
    with Image.open(Path(image_path).expanduser()) as source:
        image = source.convert("RGB")
    with Image.open(Path(mask_path).expanduser()) as source:
        mask = source.convert("RGB")  # Also supports palette PNGs and grayscale IDs.
    return _prepare_pair_images(image, mask, patch_size, long_side)


def observed_mixed_crop(image_path, mask_path, color_a, color_b,
                        padding=MIXED_CROP_PADDING):
    """Return one real image crop enclosing both selected COCO instances."""
    with Image.open(Path(image_path).expanduser()) as source:
        image = source.convert("RGB")
    with Image.open(Path(mask_path).expanduser()) as source:
        mask_image = source.convert("RGB")
    if image.size != mask_image.size:
        raise ValueError("Displayed COCO image and generated mask dimensions differ")
    mask = np.asarray(mask_image)
    selected = np.all(mask == color_a, axis=-1) | np.all(mask == color_b, axis=-1)
    if not selected.any():
        raise ValueError("The selected A+B instances have no mask pixels")
    y, x = np.nonzero(selected)
    width, height = image.size
    object_width, object_height = x.max() - x.min() + 1, y.max() - y.min() + 1
    pad_x = round(object_width * padding)
    pad_y = round(object_height * padding)
    box = (
        max(0, int(x.min()) - pad_x),
        max(0, int(y.min()) - pad_y),
        min(width, int(x.max()) + 1 + pad_x),
        min(height, int(y.max()) + 1 + pad_y),
    )
    return image.crop(box), mask_image.crop(box), box


def real_patch_mask(geometry, patch_size, threshold=PATCH_PURITY):
    """Exclude artificial right/bottom padding from a whole-crop mean."""
    resized_width, resized_height = geometry["resized_size"]
    padded_width, padded_height = geometry["padded_size"]
    rows, columns = padded_height // patch_size, padded_width // patch_size
    x0 = np.arange(columns) * patch_size
    y0 = np.arange(rows) * patch_size
    widths = np.clip(resized_width - x0, 0, patch_size)
    heights = np.clip(resized_height - y0, 0, patch_size)
    coverage = np.outer(heights, widths) / float(patch_size * patch_size)
    return coverage.reshape(-1) >= threshold


def select_patches(mask, color, patch_size, purity=PATCH_PURITY):
    h, w = mask.shape[:2]
    if h % patch_size or w % patch_size:
        raise ValueError("Mask must align with the patch grid")
    pixels = np.all(mask == color, axis=-1)
    fractions = pixels.reshape(h // patch_size, patch_size, w // patch_size, patch_size).mean(axis=(1, 3))
    return (fractions >= purity).reshape(-1), fractions.reshape(-1)


def load_teacher(path):
    backbone, metadata = load_backbone(path, "teacher", "auto")
    checkpoint = _torch_load(path)
    head, prototypes = _build_teacher_head(
        backbone, checkpoint, _teacher_head_state(_teacher_state(checkpoint))
    )
    mode = NORMALIZATION_OVERRIDE or _checkpoint_argument(checkpoint, "region_normalization", "softmax")
    default_temperature = (
        _checkpoint_argument(checkpoint, "teacher_patch_temp", .07)
        if mode == "centering"
        else _checkpoint_argument(checkpoint, "region_temp", .1)
    )
    temp = TEMPERATURE_OVERRIDE if TEMPERATURE_OVERRIDE is not None else default_temperature
    if mode not in ("centering", "softmax", "sinkhorn"):
        raise ValueError(
            f"{mode!r} is not a probability representation: 100% mass bars are undefined for signed raw_logits. "
            "Use a probability-mode checkpoint or explicitly set NORMALIZATION_OVERRIDE='softmax' "
            "to visualize a different, documented representation."
        )
    if not math.isfinite(float(temp)) or float(temp) <= 0:
        raise ValueError("Temperature must be finite and positive")
    center = None
    if mode == "centering":
        loss_state = checkpoint.get("ibot_loss", {})
        center = loss_state.get("center2") if isinstance(loss_state, dict) else None
        if center is None or center.shape != (1, 1, prototypes):
            raise ValueError(
                "Centering visualization requires checkpoint ibot_loss.center2"
            )
        center = center.detach().float().reshape(1, prototypes).cpu()
    backbone = backbone.to(DEVICE).eval().requires_grad_(False)
    head = head.to(DEVICE).eval().requires_grad_(False)
    return backbone, head, metadata, prototypes, mode, float(temp), center


@torch.inference_mode()
def _extract_prepared_sample(backbone, head, canvas, labels, geometry, patch_size):
    tensor = T.functional.to_tensor(canvas)
    tensor = T.functional.normalize(tensor, (0.485, .456, .406), (.229, .224, .225))
    tokens = backbone(tensor[None].to(DEVICE), return_all_tokens=True)
    spatial = tokens[:, _num_special_tokens(backbone):]
    _, logits = head(torch.cat((tokens[:, :1], spatial), dim=1))
    grid = (canvas.height // patch_size, canvas.width // patch_size)
    if logits.shape[1] != grid[0] * grid[1]:
        raise ValueError("Teacher patch output does not match the image grid")
    return Sample(canvas, labels, grid, logits[0].float().cpu(), geometry)


@torch.inference_mode()
def extract_sample(backbone, head, image, mask, patch_size):
    return _extract_prepared_sample(
        backbone, head, *prepare_pair(image, mask, patch_size), patch_size
    )


@torch.inference_mode()
def extract_pil_sample(backbone, head, image, mask, patch_size):
    return _extract_prepared_sample(
        backbone, head,
        *_prepare_pair_images(image, mask, patch_size),
        patch_size,
    )


@torch.inference_mode()
def normalize_bank(logits, mode, temperature, center=None):
    if logits.ndim != 2 or not len(logits) or not torch.isfinite(logits).all():
        raise ValueError("Expected nonempty finite [patches, prototypes] logits")
    if mode == "centering":
        if center is None or tuple(center.shape) != (1, logits.shape[1]):
            raise ValueError("Centering requires one center vector per prototype")
        result = ((logits.float() - center.float()) / temperature).softmax(-1)
    elif mode == "softmax":
        result = (logits.float() / temperature).softmax(-1)
    elif mode == "sinkhorn":
        result = sinkhorn_log_probabilities(logits, temperature).exp()
    else:
        raise ValueError("Mass visualizations require centering, softmax, or sinkhorn probabilities")
    return result.double().numpy()


def validate_distribution(distribution, name="distribution"):
    x = np.asarray(distribution, dtype=np.float64)
    if (x.ndim != 1 or not np.isfinite(x).all() or np.any(x < -1e-10)
            or not np.isclose(x.sum(), 1, atol=2e-5)):
        raise ValueError(f"{name} must be one finite nonnegative probability vector")
    return np.clip(x, 0, None) / x.sum()


def discriminative_prototypes(mu_a, mu_b, count=HEATMAP_PROTOTYPES):
    """Select dimensions for display only, ranked by absolute A/B difference."""
    a, b = validate_distribution(mu_a, "mu_a"), validate_distribution(mu_b, "mu_b")
    if a.shape != b.shape:
        raise ValueError("mu_a and mu_b must have the same dimensionality")
    count = min(int(count), len(a))
    delta = a - b
    indices = np.argsort(-np.abs(delta), kind="stable")[:count]
    return indices, np.sign(delta[indices]).astype(np.int8)


def concentration_diagnostics(distribution, ks=CONCENTRATION_K):
    """Top-K mass and entropy-effective support of a full fingerprint."""
    x = validate_distribution(distribution)
    ordered = np.sort(x)[::-1]
    topk = {int(k): float(ordered[:min(int(k), len(x))].sum()) for k in ks}
    positive = x[x > 0]
    effective = float(np.exp(-(positive * np.log(positive)).sum()))
    return {"topk_mass": topk, "effective_prototypes": effective}


def nonnegative_fingerprint_fit(target, mu_a, mu_b):
    """Fit target ~= alpha*mu_a + beta*mu_b with alpha,beta >= 0.

    The displayed shares use L1 magnitudes: alpha, beta, and ||residual||_1,
    normalized to sum to one. This keeps the stacked bar interpretable while
    retaining the signed full-dimensional residual for saved diagnostics.
    """
    target = validate_distribution(target, "target")
    mu_a = validate_distribution(mu_a, "mu_a")
    mu_b = validate_distribution(mu_b, "mu_b")
    if not (target.shape == mu_a.shape == mu_b.shape):
        raise ValueError("Target and fingerprints must have matching dimensions")
    design = np.stack((mu_a, mu_b), axis=1)
    unconstrained = np.linalg.lstsq(design, target, rcond=None)[0]
    candidates = [np.zeros(2)]
    candidates.append(np.array([
        max(0.0, float(mu_a @ target) / float(mu_a @ mu_a)), 0.0
    ]))
    candidates.append(np.array([
        0.0, max(0.0, float(mu_b @ target) / float(mu_b @ mu_b))
    ]))
    if np.all(unconstrained >= 0):
        candidates.append(unconstrained)
    coefficients = min(
        candidates, key=lambda c: np.square(target - design @ c).sum()
    )
    reconstruction = design @ coefficients
    residual = target - reconstruction
    residual_l1 = float(np.abs(residual).sum())
    magnitudes = np.array([coefficients[0], coefficients[1], residual_l1])
    shares = magnitudes / magnitudes.sum() if magnitudes.sum() else np.array([0., 0., 1.])
    denominator = np.linalg.norm(target) * np.linalg.norm(reconstruction)
    cosine = float(target @ reconstruction / denominator) if denominator else 0.0
    return {
        "coefficients": coefficients,
        "reconstruction": reconstruction,
        "residual": residual,
        "residual_l1": residual_l1,
        "shares": shares,
        "cosine": cosine,
    }


def leave_one_out_control(reference_means, other_mu, own_is_a):
    """Average fits of pure regions against fingerprints that exclude themselves."""
    references = np.asarray(reference_means, dtype=np.float64)
    if references.ndim != 2 or len(references) < 2:
        raise ValueError("Leave-one-out controls require at least two references")
    fits = []
    for index, target in enumerate(references):
        own = np.delete(references, index, axis=0).mean(0)
        fits.append(nonnegative_fingerprint_fit(
            target, own if own_is_a else other_mu, other_mu if own_is_a else own
        ))
    return fits


def build_composition(reference_means_a, reference_means_b, mixed,
                      sanity_probabilities, indices_a, indices_b,
                      max_mixture_patches=MIXTURE_PATCHES, seed=SEED,
                      heatmap_prototypes=HEATMAP_PROTOTYPES):
    """Build full-dimensional evidence plus a separately labeled sanity check."""
    reference_means_a = np.asarray(reference_means_a, dtype=np.float64)
    reference_means_b = np.asarray(reference_means_b, dtype=np.float64)
    if reference_means_a.ndim != 2 or reference_means_b.ndim != 2:
        raise ValueError("Reference means must have shape [regions, prototypes]")
    if min(len(reference_means_a), len(reference_means_b)) < 2:
        raise ValueError("Need at least two independent pure regions per concept")
    mu_a, mu_b = reference_means_a.mean(0), reference_means_b.mean(0)
    mixed = validate_distribution(mixed, "observed mixed crop")
    heatmap_indices, heatmap_signs = discriminative_prototypes(
        mu_a, mu_b, heatmap_prototypes
    )
    control_a = leave_one_out_control(reference_means_a, mu_b, True)
    control_b = leave_one_out_control(reference_means_b, mu_a, False)
    mixed_fit = nonnegative_fingerprint_fit(mixed, mu_a, mu_b)
    fit_groups = (control_a, control_b, [mixed_fit])
    fit_shares = np.stack([
        np.mean([fit["shares"] for fit in group], axis=0)
        for group in fit_groups
    ])
    fit_coefficients = np.stack([
        np.mean([fit["coefficients"] for fit in group], axis=0)
        for group in fit_groups
    ])
    fit_residual_l1 = np.array([
        np.mean([fit["residual_l1"] for fit in group]) for group in fit_groups
    ])
    fit_cosine = np.array([
        np.mean([fit["cosine"] for fit in group]) for group in fit_groups
    ])

    na, nb = len(indices_a), len(indices_b)
    if np.intersect1d(indices_a, indices_b).size:
        raise ValueError("Accepted A/B patch sets must be disjoint")
    if min(na, nb) < 4:
        raise ValueError(f"Need >=4 pure patches per object for exact 25% increments; found A={na}, B={nb}. Increase resolution or inspect masks/purity")
    if sanity_probabilities.shape[0] != na + nb:
        raise ValueError("Patch probability bank and selections differ")
    pa, pb = sanity_probabilities[:na], sanity_probabilities[na:]
    sanity_union = sanity_probabilities.mean(0)
    identity_error = float(np.abs(
        sanity_union - (na * pa.mean(0) + nb * pb.mean(0)) / (na + nb)
    ).max())
    n = 4 * (min(na, nb, max_mixture_patches) // 4)
    if n < 4:
        raise ValueError("Common mixture patch count must be at least four")
    rng = np.random.default_rng(seed)
    order_a, order_b = rng.permutation(na), rng.permutation(nb)
    mixture_counts, mixture_means, mixture_indices = [], [], []
    for numerator in (4, 3, 2, 1, 0):
        count_a = n * numerator // 4
        count_b = n - count_a
        ia, ib = order_a[:count_a], order_b[:count_b]
        mixture_counts.append((count_a, count_b))
        mixture_means.append(np.concatenate((pa[ia], pb[ib])).mean(0))
        mixture_indices.append(np.concatenate((np.asarray(indices_a)[ia], np.asarray(indices_b)[ib])))
    mixture_means = np.stack(mixture_means)
    mixture_fit_shares = np.stack([
        nonnegative_fingerprint_fit(row, mu_a, mu_b)["shares"]
        for row in mixture_means
    ])
    return Composition(
        mu_a, mu_b, mixed, reference_means_a, reference_means_b,
        heatmap_indices, heatmap_signs,
        {"A": concentration_diagnostics(mu_a),
         "B": concentration_diagnostics(mu_b)},
        fit_shares, fit_coefficients, fit_residual_l1, fit_cosine,
        mixed_fit["reconstruction"], mixed_fit["residual"],
        mixture_means, mixture_fit_shares, mixture_counts, mixture_indices,
        np.asarray(indices_a), np.asarray(indices_b), (na, nb), identity_error,
    )


def _panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=12, fontweight="bold", pad=12)


def _patch_overlay(ax, sample, categories, category_labels=("A", "B", "R")):
    ax.imshow(sample.image)
    gh, gw = sample.grid
    size = sample.image.width / gw
    for index, category in enumerate(categories):
        y, x = divmod(index, gw)
        if category >= 0:
            ax.add_patch(Rectangle((x * size, y * size), size, size,
                                  facecolor=COLORS[category], edgecolor="white", lw=.4, alpha=.65))
            if max(gh, gw) <= 40:
                ax.text((x + .5) * size, (y + .5) * size, category_labels[category],
                        ha="center", va="center", fontsize=5.5, color="white", weight="bold")
        else:
            ax.add_patch(Rectangle((x * size, y * size), size, size,
                                  facecolor="none", edgecolor="white", lw=.2, alpha=.25))
    ax.set_axis_off()


def _stacked_bars(ax, values, labels, xlabel="Relative L1 magnitude (%)"):
    left = np.zeros(len(values))
    for group, color in enumerate(COLORS):
        widths = 100 * values[:, group]
        ax.barh(np.arange(len(values)), widths, left=left, color=color, height=.62,
                edgecolor="white", linewidth=.7)
        for row, width in enumerate(widths):
            if width >= 8:
                ax.text(left[row] + width / 2, row, f"{width:.0f}%", ha="center", va="center",
                        fontsize=9, color="white" if group < 2 else "#202630")
        left += widths
    ax.set_yticks(np.arange(len(values)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def render_figure(source_sample, mixed_sample, mixed_patch_mask,
                  composition, protocol, path):
    c = composition
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlecolor": "#1E293B", "text.color": "#1E293B"}):
        fig = plt.figure(figsize=(17, 12.5), facecolor="white")
        grid = fig.add_gridspec(3, 3, left=.07, right=.97, top=.88, bottom=.145,
                               wspace=.38, hspace=.60, height_ratios=(1.2, .9, 1.0))
        ax = fig.add_subplot(grid[0, 0])
        ax.imshow(source_sample.image)
        for name, color in zip(("A", "B"), COLORS):
            pixels = np.all(
                source_sample.mask == protocol["mask_colors"][name], axis=-1
            )
            if pixels.any():
                ax.contour(pixels.astype(float), levels=[.5], colors=[color], linewidths=1.5)
                y, x = np.nonzero(pixels)
                ax.text(x.mean(), y.mean(), name, ha="center", va="center", weight="bold",
                        color="white", bbox={"facecolor": color, "edgecolor": "white", "boxstyle": "round"})
        _panel_title(ax, "a", f"Source image  |  {CONCEPT_A} + {CONCEPT_B}")
        ax.set_axis_off()
        ax = fig.add_subplot(grid[0, 1])
        used = np.where(mixed_patch_mask, 2, -1)
        _patch_overlay(ax, mixed_sample, used)
        _panel_title(ax, "b", "Independently observed A+B crop")
        ax.text(
            .5, -.09,
            f"Separate model forward · mean of {mixed_patch_mask.sum()} real crop patches",
            transform=ax.transAxes, ha="center", fontsize=9,
        )
        ax = fig.add_subplot(grid[0, 2])
        requested_k = list(CONCENTRATION_K)
        x = np.arange(len(requested_k))
        width = .36
        for offset, concept, color in ((-.5, "A", COLORS[0]), (.5, "B", COLORS[1])):
            values = [100 * c.concentration[concept]["topk_mass"][k]
                      for k in requested_k]
            ax.bar(x + offset * width, values, width, color=color,
                   label=f"{concept} fingerprint")
        ax.set_xticks(x, [f"top-{k}" for k in requested_k])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Captured probability mass (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.text(
            .02, .97,
            "entropy-effective prototypes\n"
            f"A: {c.concentration['A']['effective_prototypes']:.1f}   "
            f"B: {c.concentration['B']['effective_prototypes']:.1f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#CBD5E1"},
        )
        _panel_title(ax, "c", "Fingerprint concentration")

        ax = fig.add_subplot(grid[1, :2])
        indices = c.heatmap_indices
        fingerprint = np.stack((c.mu_a, c.mu_b, c.mixed))[:, indices] * 100
        image = ax.imshow(fingerprint, cmap="Blues", vmin=0, vmax=max(float(fingerprint.max()), 1e-6), aspect="auto")
        ax.set_yticks(range(3), [r"$\mu_A$ references", r"$\mu_B$ references", "Observed A+B crop"])
        ax.set_xticks(range(len(indices)), [f"p{k}" for k in indices], rotation=45, ha="right", fontsize=8)
        for sign, tick in zip(c.heatmap_signs, ax.get_xticklabels()):
            tick.set_color(COLORS[0 if sign >= 0 else 1])
        _panel_title(ax, "d", "Most discriminative prototype dimensions")
        fig.colorbar(image, ax=ax, fraction=.025, pad=.025, label="Probability (%)")
        ax.set_xlabel(
            "Selected by |μA−μB| for illustration only · decomposition uses every prototype dimension"
        )

        ax = fig.add_subplot(grid[1, 2])
        _stacked_bars(
            ax, c.fit_shares,
            ["Pure A\nleave-one-out", "Pure B\nleave-one-out", "Observed A+B\nseparate forward"],
        )
        _panel_title(ax, "e", "Full-dimensional fingerprint decomposition")
        ax.text(
            .5, -.43,
            "NNLS: r ≈ αμA + βμB + ε\n"
            f"mixed α={c.fit_coefficients[2, 0]:.3f}, "
            f"β={c.fit_coefficients[2, 1]:.3f}, "
            f"||ε||₁={c.fit_residual_l1[2]:.3f}, "
            f"cos={c.fit_cosine[2]:.3f}",
            transform=ax.transAxes, ha="center", fontsize=8.5,
        )

        ax = fig.add_subplot(grid[2, :2])
        labels = [f"{a / (a+b):.0%} A\n{a}A + {b}B" for a, b in c.mixture_counts]
        _stacked_bars(ax, c.mixture_fit_shares, labels)
        _panel_title(ax, "f", "Sanity check only: controlled patch-count mixtures")
        ax.text(.5, -.32, "Mean-pooling progression is mathematically guaranteed; it is not composition evidence.",
                transform=ax.transAxes, ha="center", fontsize=9)

        ax = fig.add_subplot(grid[2, 2])
        ax.set_axis_off()
        _panel_title(ax, "g", "Composition test protocol")
        lines = [r"$\mu_A=\mathrm{mean}(r_A^{ref})$", r"$\mu_B=\mathrm{mean}(r_B^{ref})$", "",
                 r"$r_{AB}^{obs}\approx\alpha\mu_A+\beta\mu_B+\epsilon$", "",
                 f"Independent calibration: {protocol['reference_counts'][0]} A / {protocol['reference_counts'][1]} B regions",
                 f"Full fingerprint: {len(c.mu_a)} prototype dimensions",
                 f"Heatmap illustration: {len(c.heatmap_indices)} dimensions",
                 f"Teacher patch head · {protocol['mode']} · T={protocol['temperature']:g}",
                 f"Sanity identity residual: {c.identity_error:.1e}"]
        ax.text(0, .92, "\n".join(lines), va="top", fontsize=10, linespacing=1.6)
        fig.suptitle("Controlled A + B composition in learned prototype space", fontsize=19, weight="bold", y=.97)
        fig.text(.5, .927, f"{CONCEPT_A} (A)  +  {CONCEPT_B} (B)   ·   EMA teacher   ·   No PCA, UMAP or learned projection",
                 ha="center", fontsize=11)
        fig.legend(handles=[Patch(color=color, label=name) for color, name in zip(
            COLORS, ("A fingerprint contribution", "B fingerprint contribution", "Residual magnitude"))],
            loc="lower center", bbox_to_anchor=(.5, .038), ncol=3, frameon=False)
        fig.text(.5, .015, "Primary evidence: a separately forwarded real A+B crop fitted with full-dimensional independent fingerprints. Panel f is only an arithmetic sanity check.",
                 ha="center", fontsize=9, color="#586575")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, facecolor="white")
        plt.close(fig)


def main():
    checkpoint = Path(CHECKPOINT).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Configure CHECKPOINT at the top of the script: {checkpoint}")
    image, mask, coco = find_coco_pair(COCO_IMAGE_ID, DATASETS_ROOT)
    if bool(REFERENCE_A) != bool(REFERENCE_B):
        raise ValueError("Configure both REFERENCE_A and REFERENCE_B, or leave both empty for automatic COCO references")
    if REFERENCE_A:
        references_a, references_b = REFERENCE_A, REFERENCE_B
        reference_source = "configured"
    else:
        references_a, references_b = find_coco_references(coco)
        reference_source = "automatic_coco"
        print(
            f"Selected {len(references_a)} A and {len(references_b)} B "
            "independent references from COCO",
            flush=True,
        )
    validate_inputs(checkpoint, image, mask, references_a, references_b)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    backbone, head, metadata, prototypes, mode, temperature, center = load_teacher(checkpoint)
    patch_size = int(metadata["patch_size"])
    source_sample = extract_sample(backbone, head, image, mask, patch_size)
    color_a, color_b = resolve_colors(source_sample.mask)
    print(f"Mask mapping: A={CONCEPT_A} RGB{color_a}; B={CONCEPT_B} RGB{color_b}", flush=True)
    selected_a, fractions_a = select_patches(source_sample.mask, color_a, patch_size)
    selected_b, fractions_b = select_patches(source_sample.mask, color_b, patch_size)
    ia, ib = np.flatnonzero(selected_a), np.flatnonzero(selected_b)
    if min(len(ia), len(ib)) < 4:
        raise ValueError(f"Not enough pure patches for controlled mixtures: A={len(ia)}, B={len(ib)}; increase LONG_SIDE or inspect the masks")

    mixed_image, mixed_mask, mixed_box = observed_mixed_crop(
        image, mask, color_a, color_b
    )
    mixed_sample = extract_pil_sample(
        backbone, head, mixed_image, mixed_mask, patch_size
    )
    mixed_patch_mask = real_patch_mask(mixed_sample.geometry, patch_size)
    if not mixed_patch_mask.any():
        raise ValueError("The observed A+B crop has no complete real-image patches")

    reference_banks, reference_records = [], []
    for concept, references in (("A", references_a), ("B", references_b)):
        for reference in references:
            ref_sample = extract_sample(backbone, head, reference.image, reference.mask, patch_size)
            selected, _ = select_patches(ref_sample.mask, reference.color, patch_size)
            if not selected.any():
                raise ValueError(f"No pure patches in reference {reference.image} for color {reference.color}")
            reference_banks.append(ref_sample.logits[selected])
            reference_records.append({"concept": concept, "image": str(Path(reference.image).expanduser().resolve()),
                                      "mask": str(Path(reference.mask).expanduser().resolve()), "color": reference.color,
                                      "accepted_patch_indices": np.flatnonzero(selected).tolist(), "geometry": ref_sample.geometry})
            print(f"Calibration {concept}: {reference.image}: {int(selected.sum())} pure patches", flush=True)
    # One calibration assignment bank across independent A/B references. SK is
    # not recalculated per pure region (that would balance each region itself).
    reference_probabilities = normalize_bank(
        torch.cat(reference_banks), mode, temperature, center
    )
    region_reference_means = []
    offset = 0
    for bank in reference_banks:
        region_reference_means.append(reference_probabilities[offset:offset + len(bank)].mean(0))
        offset += len(bank)
    reference_means_a = np.stack(region_reference_means[:len(references_a)])
    reference_means_b = np.stack(region_reference_means[len(references_a):])

    # The primary A+B representation comes from its own real crop and model
    # forward. It is never assembled from A/B segmentation-selected patches.
    mixed_probabilities = normalize_bank(
        mixed_sample.logits[mixed_patch_mask], mode, temperature, center
    )
    mixed_representation = mixed_probabilities.mean(0)

    # Segmentation-selected patches from the source image are retained only for
    # the explicitly labeled arithmetic sanity check in panel f.
    sanity_probabilities = normalize_bank(
        source_sample.logits[np.concatenate((ia, ib))], mode, temperature, center
    )
    result = build_composition(
        reference_means_a, reference_means_b, mixed_representation,
        sanity_probabilities, ia, ib,
    )
    protocol = {"checkpoint": str(checkpoint), "image": str(image), "mask": str(mask),
                "coco": coco, "mode": mode, "temperature": temperature, "normalization_override": NORMALIZATION_OVERRIDE,
                "temperature_override": TEMPERATURE_OVERRIDE,
                "teacher_center_applied": mode == "centering",
                "reference_counts": [len(references_a), len(references_b)],
                "reference_source": reference_source,
                "references": reference_records,
                "mask_colors": {"A": color_a, "B": color_b, "background": BACKGROUND_COLOR},
                "concept_names": [CONCEPT_A, CONCEPT_B], "metadata": metadata,
                "source_geometry": source_sample.geometry, "source_grid": source_sample.grid,
                "mixed_crop": {"original_box": mixed_box,
                               "padding_fraction": MIXED_CROP_PADDING,
                               "geometry": mixed_sample.geometry,
                               "grid": mixed_sample.grid,
                               "used_patch_indices": np.flatnonzero(mixed_patch_mask).tolist()},
                "purity": PATCH_PURITY,
                "image_digest": image_digest(image),
                "seed": SEED, "prototypes": prototypes,
                "full_dimensional_fingerprints": True,
                "heatmap_prototypes_requested": HEATMAP_PROTOTYPES,
                "heatmap_prototype_indices": result.heatmap_indices.tolist(),
                "concentration": result.concentration,
                "decomposition": {
                    "method": "nonnegative least squares over all prototype dimensions",
                    "bar_definition": "normalized [alpha, beta, L1 residual magnitude]",
                    "labels": ["pure_A_leave_one_out", "pure_B_leave_one_out", "observed_A+B"],
                    "shares": result.fit_shares.tolist(),
                    "coefficients": result.fit_coefficients.tolist(),
                    "residual_l1": result.fit_residual_l1.tolist(),
                    "cosine_to_reconstruction": result.fit_cosine.tolist(),
                },
                "counts": result.counts, "mixture_counts": result.mixture_counts,
                "mixture_indices": [x.tolist() for x in result.mixture_indices],
                "mixture_fit_shares": result.mixture_fit_shares.tolist(),
                "identity_error": result.identity_error,
                "interpretation": "Primary test uses an independently observed A+B crop; controlled patch mixtures are a sanity check only",
                "sinkhorn_scope": "Independent reference bank; observed mixed-crop bank; source-image sanity bank"}
    stem = f"{checkpoint.stem}_{image.stem}_composition"
    path = Path(OUTPUT_DIR) / f"{stem}.png"
    render_figure(
        source_sample, mixed_sample, mixed_patch_mask, result, protocol, path
    )
    np.savez_compressed(
        path.with_suffix(".npz"),
        mu_a=result.mu_a, mu_b=result.mu_b,
        reference_means_a=result.reference_means_a,
        reference_means_b=result.reference_means_b,
        observed_mixed_representation=result.mixed,
        mixed_patch_probabilities=mixed_probabilities,
        mixed_patch_indices=np.flatnonzero(mixed_patch_mask),
        fit_shares=result.fit_shares,
        fit_coefficients=result.fit_coefficients,
        fit_residual_l1=result.fit_residual_l1,
        fit_cosine=result.fit_cosine,
        mixed_reconstruction=result.mixed_reconstruction,
        mixed_residual=result.mixed_residual,
        mixture_means=result.mixture_means,
        mixture_fit_shares=result.mixture_fit_shares,
        sanity_patch_probabilities=sanity_probabilities,
        sanity_patch_indices=np.concatenate((ia, ib)),
        fractions_a=fractions_a, fractions_b=fractions_b,
    )
    path.with_suffix(".json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}\nFull prototype vectors: {path.with_suffix('.npz')}\nProtocol: {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
