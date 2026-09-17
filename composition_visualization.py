#!/usr/bin/env python3
"""Paper figure of A+B patch-distribution composition, without projection.

Usage: python composition_visualization.py
Edit the configuration below, especially independent REFERENCE_A/B regions.
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
TOP_K = 8  # Up to K strictly positive/negative contrast components, disjoint.
MIXTURE_PATCHES = 64  # Common count, reduced to available patches; multiple of 4.
SEED = 0
DPI = 300
NORMALIZATION_OVERRIDE = None  # None uses checkpoint region_normalization.
TEMPERATURE_OVERRIDE = None  # None uses checkpoint region_temp (fallback .1).
COLORS = ("#2676B8", "#D97924", "#A5ADB6")  # A-associated / B-associated / other


@dataclass(frozen=True)
class ReferenceRegion:
    image: str | Path
    mask: str | Path
    color: tuple[int, int, int]


# Independent images, not the displayed image or copies of it. Each entry
# defines ONE pure object region; masks may contain additional unrelated colors.
# Multiple regions are averaged equally, irrespective of their patch counts.
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
    prototypes_a: np.ndarray
    prototypes_b: np.ndarray
    means: np.ndarray  # A, B, union; full K dimensions
    mixture_means: np.ndarray
    masses: np.ndarray
    mixture_masses: np.ndarray
    mixture_counts: list[tuple[int, int]]
    mixture_indices: list[np.ndarray]
    selected_a: np.ndarray
    selected_b: np.ndarray
    patch_masses: np.ndarray  # [all patches, 3], unselected entries NaN
    counts: tuple[int, int]
    identity_error: float


def find_coco_pair(image_id, datasets_root):
    """Find standard COCO instances annotations and rasterize two object masks."""
    root = Path(datasets_root).expanduser().resolve()
    if image_id < 0 or not root.is_dir():
        raise ValueError("Require a nonnegative COCO image ID and an existing datasets root")
    matches = []
    for path in sorted(root.rglob("instances_*.json")):
        with path.open() as stream:
            data = json.load(stream)
        records = [r for r in data.get("images", []) if r["id"] == image_id]
        if records:
            matches.append((path, records[0], data))
    if not matches:
        raise FileNotFoundError(f"COCO image {image_id} not found in instances_*.json below {root}; extract COCO images and instance annotations first")
    if len(matches) != 1:
        raise ValueError(f"Image {image_id} occurs in multiple annotation files: {[str(x[0]) for x in matches]}; use a narrower datasets root")
    annotation_path, record, data = matches[0]
    filename = Path(record["file_name"]).name
    images = sorted(p for p in root.rglob(filename) if p.is_file())
    if len(images) != 1:
        raise ValueError(f"Expected one local {filename} below {root}, found {len(images)}; extract images or use a narrower root")
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
    def decode(annotation):
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
    ma, mb = decode(a), decode(b)
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
    provenance = {"image_id": image_id, "datasets_root": str(root), "annotations": str(annotation_path),
                  "objects": [{"annotation_id": obj["id"], "category_id": obj["category_id"],
                               "category": categories[obj["category_id"]]} for obj in (a, b)],
                  "ambiguous_pixels_excluded": int(ambiguous.sum())}
    print(f"COCO {image_id}: A={provenance['objects'][0]}, B={provenance['objects'][1]}", flush=True)
    return images[0], path, provenance


def image_digest(path):
    """Detect byte-identical decoded images even if copied/re-encoded losslessly."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def validate_inputs(checkpoint, image, mask):
    if not REFERENCE_A or not REFERENCE_B:
        raise ValueError("Configure independent REFERENCE_A and REFERENCE_B image/mask/color entries at the top of the script")
    if not .5 < PATCH_PURITY <= 1 or TOP_K < 1 or MIXTURE_PATCHES < 4 or LONG_SIDE < 1:
        raise ValueError("Require .5 < PATCH_PURITY <= 1, TOP_K >= 1, MIXTURE_PATCHES >= 4, LONG_SIDE >= 1")
    paths = [checkpoint, image, mask]
    for reference in (*REFERENCE_A, *REFERENCE_B):
        paths.extend((reference.image, reference.mask))
        if len(reference.color) != 3 or any(not 0 <= value <= 255 for value in reference.color):
            raise ValueError("Reference colors must be RGB triples in [0, 255]")
    for path in paths:
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(path)
    target_digest = image_digest(image)
    for reference in (*REFERENCE_A, *REFERENCE_B):
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


def prepare_pair(image_path, mask_path, patch_size, long_side=None):
    """Resize both together, then pad to a patch grid without removing objects."""
    with Image.open(Path(image_path).expanduser()) as source:
        image = source.convert("RGB")
    with Image.open(Path(mask_path).expanduser()) as source:
        mask = source.convert("RGB")  # Also supports palette PNGs and grayscale IDs.
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
def extract_sample(backbone, head, image, mask, patch_size):
    canvas, labels, geometry = prepare_pair(image, mask, patch_size)
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


def associated_sets(mu_a, mu_b, top_k=TOP_K):
    """Disjoint positive/negative contrast sets; exact ties belong to other."""
    delta = np.asarray(mu_a) - np.asarray(mu_b)
    order_a = np.argsort(-delta, kind="stable")
    order_b = np.argsort(delta, kind="stable")
    a = order_a[delta[order_a] > 1e-12][:top_k]
    b = order_b[delta[order_b] < -1e-12][:top_k]
    if not len(a) or not len(b):
        raise ValueError("Independent references yield no contrasting A/B components; inspect calibration rather than forcing prototype labels")
    return a, b


def component_mass(distributions, a, b):
    x = np.asarray(distributions, dtype=np.float64)
    if np.intersect1d(a, b).size:
        raise ValueError("A/B component sets must be disjoint")
    if (not np.isfinite(x).all() or np.any(x < 0)
            or not np.allclose(x.sum(-1), 1, atol=2e-5)):
        raise ValueError("Expected normalized nonnegative probabilities")
    other = np.ones(x.shape[-1], dtype=bool)
    other[a] = False
    other[b] = False
    return np.stack((x[..., a].sum(-1), x[..., b].sum(-1), x[..., other].sum(-1)), axis=-1)


def build_composition(probabilities, indices_a, indices_b, mu_a, mu_b,
                      total_patches, max_mixture_patches=MIXTURE_PATCHES, seed=SEED, top_k=TOP_K):
    """Probabilities are frozen once; rows correspond to A patches then B patches."""
    na, nb = len(indices_a), len(indices_b)
    if np.intersect1d(indices_a, indices_b).size:
        raise ValueError("Accepted A/B patch sets must be disjoint")
    if min(na, nb) < 4:
        raise ValueError(f"Need >=4 pure patches per object for exact 25% increments; found A={na}, B={nb}. Increase resolution or inspect masks/purity")
    if probabilities.shape[0] != na + nb:
        raise ValueError("Patch probability bank and selections differ")
    a, b = associated_sets(mu_a, mu_b, top_k)
    pa, pb = probabilities[:na], probabilities[na:]
    means = np.stack((pa.mean(0), pb.mean(0), probabilities.mean(0)))
    identity_error = float(np.abs(means[2] - (na * means[0] + nb * means[1]) / (na + nb)).max())
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
    patch_masses = np.full((total_patches, 3), np.nan)
    patch_masses[np.concatenate((indices_a, indices_b))] = component_mass(probabilities, a, b)
    return Composition(a, b, means, mixture_means, component_mass(means, a, b),
                       component_mass(mixture_means, a, b), mixture_counts, mixture_indices,
                       np.asarray(indices_a), np.asarray(indices_b), patch_masses, (na, nb), identity_error)


def _panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=12, fontweight="bold", pad=12)


def _patch_overlay(ax, sample, categories):
    ax.imshow(sample.image)
    gh, gw = sample.grid
    size = sample.image.width / gw
    for index, category in enumerate(categories):
        y, x = divmod(index, gw)
        if category >= 0:
            ax.add_patch(Rectangle((x * size, y * size), size, size,
                                  facecolor=COLORS[category], edgecolor="white", lw=.4, alpha=.65))
            if max(gh, gw) <= 40:
                ax.text((x + .5) * size, (y + .5) * size, "ABO"[category],
                        ha="center", va="center", fontsize=5.5, color="white", weight="bold")
        else:
            ax.add_patch(Rectangle((x * size, y * size), size, size,
                                  facecolor="none", edgecolor="white", lw=.2, alpha=.25))
    ax.set_axis_off()


def _stacked_bars(ax, values, labels):
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
    ax.set_xlabel("Probability mass (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def render_figure(sample, composition, protocol, path):
    c = composition
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlecolor": "#1E293B", "text.color": "#1E293B"}):
        fig = plt.figure(figsize=(17, 12.5), facecolor="white")
        grid = fig.add_gridspec(3, 3, left=.07, right=.97, top=.88, bottom=.145,
                               wspace=.38, hspace=.60, height_ratios=(1.2, .9, 1.0))
        ax = fig.add_subplot(grid[0, 0])
        ax.imshow(sample.image)
        for name, color in zip(("A", "B"), COLORS):
            pixels = np.all(sample.mask == protocol["mask_colors"][name], axis=-1)
            if pixels.any():
                ax.contour(pixels.astype(float), levels=[.5], colors=[color], linewidths=1.5)
                y, x = np.nonzero(pixels)
                ax.text(x.mean(), y.mean(), name, ha="center", va="center", weight="bold",
                        color="white", bbox={"facecolor": color, "edgecolor": "white", "boxstyle": "round"})
        _panel_title(ax, "a", f"Source image  |  {CONCEPT_A} + {CONCEPT_B}")
        ax.set_axis_off()
        ax = fig.add_subplot(grid[0, 1])
        gt = np.full(len(sample.logits), -1)
        gt[c.selected_a], gt[c.selected_b] = 0, 1
        _patch_overlay(ax, sample, gt)
        _panel_title(ax, "b", "Mask-selected pure patches")
        ax.text(.5, -.09, f"A: {c.counts[0]} patches    B: {c.counts[1]} patches    purity ≥ {PATCH_PURITY:.0%}",
                transform=ax.transAxes, ha="center", fontsize=9)
        ax = fig.add_subplot(grid[0, 2])
        classes = np.full(len(sample.logits), -1)
        accepted = np.concatenate((c.selected_a, c.selected_b))
        classes[accepted] = c.patch_masses[accepted].argmax(-1)
        _patch_overlay(ax, sample, classes)
        _panel_title(ax, "c", "Largest component mass per patch")
        mass = c.masses[2] * 100
        ax.annotate(f"mean → A {mass[0]:.1f}%  |  B {mass[1]:.1f}%  |  other {mass[2]:.1f}%",
                    xy=(.5, 0), xytext=(.5, -.13), xycoords="axes fraction", textcoords="axes fraction",
                    ha="center", fontsize=8.5, bbox={"boxstyle": "round,pad=.4", "fc": "#F1F5F9", "ec": "#CBD5E1"})

        ax = fig.add_subplot(grid[1, :2])
        indices = np.concatenate((c.prototypes_a, c.prototypes_b))
        fingerprint = c.means[:, indices] * 100
        image = ax.imshow(fingerprint, cmap="Blues", vmin=0, vmax=max(float(fingerprint.max()), 1e-6), aspect="auto")
        ax.set_yticks(range(3), ["A only", "B only", "A + B"])
        ax.set_xticks(range(len(indices)), [f"p{k}" for k in indices], rotation=45, ha="right", fontsize=8)
        split = len(c.prototypes_a)
        ax.axvline(split - .5, color="#FFFFFF", linewidth=3)
        for j, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(COLORS[int(j >= split)])
        _panel_title(ax, "d", "Prototype fingerprints  |  A-associated then B-associated")
        fig.colorbar(image, ax=ax, fraction=.025, pad=.025, label="Probability (%)")
        ax.set_xlabel("Fixed components selected using independent reference regions; common color scale")

        ax = fig.add_subplot(grid[1, 2])
        _stacked_bars(ax, c.masses, ["A only", "B only", "A + B"])
        _panel_title(ax, "e", "Composition of the accepted regions")
        ax.text(.5, -.35, f"A+B uses all {sum(c.counts)} accepted patches", transform=ax.transAxes, ha="center", fontsize=9)

        ax = fig.add_subplot(grid[2, :2])
        labels = [f"{a / (a+b):.0%} A\n{a}A + {b}B" for a, b in c.mixture_counts]
        _stacked_bars(ax, c.mixture_masses, labels)
        _panel_title(ax, "f", "Controlled patch-count mixtures")
        ax.text(.5, -.32, "Fixed patch assignments; deterministic subsets without replacement. No image recompositing.",
                transform=ax.transAxes, ha="center", fontsize=9)

        ax = fig.add_subplot(grid[2, 2])
        ax.set_axis_off()
        _panel_title(ax, "g", "From patches to a region")
        lines = [r"$p_i$ → mean over accepted patches → $r$", "",
                 r"$r_{A+B}=\frac{n_A r_A+n_B r_B}{n_A+n_B}$", "",
                 f"Independent calibration: {protocol['reference_counts'][0]} A / {protocol['reference_counts'][1]} B regions",
                 f"Components: {len(c.prototypes_a)} A / {len(c.prototypes_b)} B (disjoint)",
                 f"Teacher raw patch head · {protocol['mode']} · T={protocol['temperature']:g}",
                 f"Arithmetic identity residual: {c.identity_error:.1e}"]
        ax.text(0, .92, "\n".join(lines), va="top", fontsize=10, linespacing=1.6)
        fig.suptitle("Controlled A + B composition in learned prototype space", fontsize=19, weight="bold", y=.97)
        fig.text(.5, .927, f"{CONCEPT_A} (A)  +  {CONCEPT_B} (B)   ·   EMA teacher   ·   No PCA, UMAP or learned projection",
                 ha="center", fontsize=11)
        fig.legend(handles=[Patch(color=color, label=name) for color, name in zip(
            COLORS, ("A-associated latent components", "B-associated latent components", "Other components"))],
            loc="lower center", bbox_to_anchor=(.5, .038), ncol=3, frameon=False)
        fig.text(.5, .015, "Patch-mixture linearity follows from mean pooling. Independent calibration tests component association, not compositional generalization.",
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
    validate_inputs(checkpoint, image, mask)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    backbone, head, metadata, prototypes, mode, temperature, center = load_teacher(checkpoint)
    patch_size = int(metadata["patch_size"])
    sample = extract_sample(backbone, head, image, mask, patch_size)
    color_a, color_b = resolve_colors(sample.mask)
    print(f"Mask mapping: A={CONCEPT_A} RGB{color_a}; B={CONCEPT_B} RGB{color_b}", flush=True)
    selected_a, fractions_a = select_patches(sample.mask, color_a, patch_size)
    selected_b, fractions_b = select_patches(sample.mask, color_b, patch_size)
    ia, ib = np.flatnonzero(selected_a), np.flatnonzero(selected_b)
    if min(len(ia), len(ib)) < 4:
        raise ValueError(f"Not enough pure patches for controlled mixtures: A={len(ia)}, B={len(ib)}; increase LONG_SIDE or inspect the masks")

    reference_banks, reference_records = [], []
    for concept, references in (("A", REFERENCE_A), ("B", REFERENCE_B)):
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
    mu_a = np.mean(region_reference_means[:len(REFERENCE_A)], axis=0)
    mu_b = np.mean(region_reference_means[len(REFERENCE_A):], axis=0)
    # Normalize the displayed union once, then freeze assignments for mixtures.
    probabilities = normalize_bank(
        sample.logits[np.concatenate((ia, ib))], mode, temperature, center
    )
    result = build_composition(probabilities, ia, ib, mu_a, mu_b, len(sample.logits))
    protocol = {"checkpoint": str(checkpoint), "image": str(image), "mask": str(mask),
                "coco": coco, "mode": mode, "temperature": temperature, "normalization_override": NORMALIZATION_OVERRIDE,
                "temperature_override": TEMPERATURE_OVERRIDE,
                "teacher_center_applied": mode == "centering",
                "reference_counts": [len(REFERENCE_A), len(REFERENCE_B)], "references": reference_records,
                "mask_colors": {"A": color_a, "B": color_b, "background": BACKGROUND_COLOR},
                "concept_names": [CONCEPT_A, CONCEPT_B], "metadata": metadata,
                "geometry": sample.geometry, "grid": sample.grid, "purity": PATCH_PURITY,
                "image_digest": image_digest(image),
                "seed": SEED, "top_k_requested": TOP_K, "prototypes": prototypes,
                "prototype_sets": {"A": result.prototypes_a.tolist(), "B": result.prototypes_b.tolist()},
                "counts": result.counts, "mixture_counts": result.mixture_counts,
                "mixture_indices": [x.tolist() for x in result.mixture_indices],
                "masses": result.masses.tolist(), "mixture_masses": result.mixture_masses.tolist(),
                "identity_error": result.identity_error,
                "interpretation": "Controlled pooling of fixed patches; arithmetic linearity is by construction",
                "sinkhorn_scope": "Independent calibration bank; target accepted A+B bank normalized once; no mixture refits"}
    stem = f"{checkpoint.stem}_{image.stem}_composition"
    path = Path(OUTPUT_DIR) / f"{stem}.png"
    render_figure(sample, result, protocol, path)
    np.savez_compressed(path.with_suffix(".npz"), mu_a=mu_a, mu_b=mu_b, region_means=result.means,
                        mixture_means=result.mixture_means, patch_probabilities=probabilities,
                        patch_indices=np.concatenate((ia, ib)), fractions_a=fractions_a, fractions_b=fractions_b)
    path.with_suffix(".json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}\nFull prototype vectors: {path.with_suffix('.npz')}\nProtocol: {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
