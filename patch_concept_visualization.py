#!/usr/bin/env python3
"""Compare two teacher patch-concept visualizations on identical ImageNet images.

The script intentionally has no command-line interface. Edit the constants in
the configuration section and run it directly. It uses the repository's
checkpoint/backbone loader and the same deterministic ImageNet preprocessing
as :mod:`pca_visualization`.
"""

from __future__ import annotations

import gc
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import datasets, transforms as T


# -----------------------------------------------------------------------------
# Repository paths and experiment settings (hard-coded by design)
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils.common import load_backbone  # noqa: E402
from losses.sinkhorn import sinkhorn_knopp  # noqa: E402
from model import iBOTHead  # noqa: E402


IMAGENET_VAL = "/mnt/fast/nobackup/scratch4weeks/jg02228/datasets"
OUTPUT_DIR = REPO_ROOT / "output" / "patch_concept_visualizations"

CHECKPOINT_1 = REPO_ROOT / "checkpoints" / "ibot_vit_small.pth"
CHECKPOINT_2 = (
    REPO_ROOT
    / "checkpoints"
    / "checkpoint_source1000_continuation0200.pth"
)
CHECKPOINT_1_NAME = "Official iBOT"
CHECKPOINT_2_NAME = "Overlap"

CHECKPOINT_KEY = "teacher"
NUM_IMAGES = 5
SEED = 0
TOP_K_CONCEPTS = 5
KMEANS_CLUSTERS = 2
KMEANS_N_INIT = 10

# Match pca_visualization.py: 560 is divisible by ViT-S/16 and gives a dense
# 35 x 35 grid while retaining a convenient high-resolution output image.
VIS_RESOLUTION = 560
IMAGE_DPI = 220

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deterministic resize/crop and model input transforms, shared by both models.
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


@dataclass
class TeacherModel:
    """Loaded teacher backbone, patch head, and target-normalization state."""

    backbone: torch.nn.Module
    head: iBOTHead
    metadata: dict
    patch_target_mode: str
    patch_temperature: float
    patch_center: torch.Tensor | None
    concepts: int


@dataclass
class ImageResult:
    """Patch assignments and probabilities for one model/image pair."""

    cluster_map: np.ndarray  # [patch_grid_h, patch_grid_w]
    object_cluster: int
    probabilities: np.ndarray  # [num_pseudo_concepts]


def _set_deterministic_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _check_inputs() -> None:
    if NUM_IMAGES < 1:
        raise ValueError("NUM_IMAGES must be positive")
    if TOP_K_CONCEPTS < 1:
        raise ValueError("TOP_K_CONCEPTS must be positive")
    if KMEANS_CLUSTERS != 2:
        raise ValueError("This visualization requires exactly two k-means clusters")
    if not Path(IMAGENET_VAL).is_dir():
        raise FileNotFoundError(
            f"ImageNet validation directory not found: {IMAGENET_VAL}"
        )
    missing = [
        str(path)
        for path in (CHECKPOINT_1, CHECKPOINT_2)
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "The following checkpoints are missing:\n  - " + "\n  - ".join(missing)
        )


def _sample_images() -> tuple[datasets.ImageFolder, list[int]]:
    """Select one deterministic validation subset shared by both checkpoints."""
    dataset = datasets.ImageFolder(IMAGENET_VAL)
    if NUM_IMAGES > len(dataset):
        raise ValueError(f"NUM_IMAGES={NUM_IMAGES} exceeds dataset size {len(dataset)}")
    indices = np.random.default_rng(SEED).choice(
        len(dataset), size=NUM_IMAGES, replace=False
    ).tolist()
    return dataset, indices


def _prepare_selected_images(
    dataset: datasets.ImageFolder, indices: list[int]
) -> tuple[list[Image.Image], list[dict]]:
    """Load and geometrically preprocess selected images exactly once."""
    images: list[Image.Image] = []
    records: list[dict] = []
    manifest = ["sample_number\tdataset_index\tclass_index\tclass_name\tpath"]
    for sample_number, index in enumerate(indices):
        image, class_index = dataset[index]
        image = GEOMETRIC_TRANSFORM(image.convert("RGB"))
        class_name = dataset.classes[class_index]
        path = Path(dataset.samples[index][0])
        images.append(image)
        records.append(
            {
                "sample_number": sample_number,
                "dataset_index": index,
                "class_index": class_index,
                "class_name": class_name,
                "path": str(path),
            }
        )
        manifest.append(
            f"{sample_number}\t{index}\t{class_index}\t{class_name}\t{path}"
        )
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT_DIR) / "selected_images.tsv").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )
    return images, records


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_argument(checkpoint, name: str, default=None):
    arguments = checkpoint.get("args") if isinstance(checkpoint, Mapping) else None
    if isinstance(arguments, Mapping):
        return arguments.get(name, default)
    return getattr(arguments, name, default)


def _teacher_state(checkpoint) -> dict[str, torch.Tensor]:
    """Extract a teacher state dict from the checkpoint layouts used here."""
    if isinstance(checkpoint, Mapping) and CHECKPOINT_KEY in checkpoint:
        state = checkpoint[CHECKPOINT_KEY]
    elif isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, Mapping) and checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        state = checkpoint
    else:
        raise ValueError(
            f"Checkpoint has no '{CHECKPOINT_KEY}' state needed for its patch head"
        )
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint entry '{CHECKPOINT_KEY}' is not a state dict")
    return {key: value for key, value in state.items() if torch.is_tensor(value)}


def _strip_state_prefix(name: str) -> str:
    """Normalize DDP/compiled teacher prefixes without hiding key mismatches."""
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod.", "teacher."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _teacher_head_state(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    head_state = {}
    for name, value in raw_state.items():
        name = _strip_state_prefix(name)
        if name.startswith("head."):
            head_state[name[len("head.") :]] = value
    if not head_state:
        raise ValueError(
            "Teacher checkpoint contains no projection-head weights; a full iBOT "
            "teacher checkpoint is required for pseudo-concept probabilities"
        )
    return head_state


def _copy_shared_head_aliases(
    state: dict[str, torch.Tensor], shared_head: bool
) -> dict[str, torch.Tensor]:
    """Accept checkpoints that serialize only one name for an aliased layer."""
    if not shared_head:
        return state
    state = dict(state)
    for first, second in (("last_layer.", "last_layer2."), ("last_norm.", "last_norm2.")):
        for name, value in list(state.items()):
            if name.startswith(first):
                state.setdefault(second + name[len(first) :], value)
            elif name.startswith(second):
                state.setdefault(first + name[len(second) :], value)
    return state


def _build_teacher_head(
    backbone: torch.nn.Module,
    checkpoint,
    head_state: dict[str, torch.Tensor],
) -> tuple[iBOTHead, int]:
    """Construct the exact iBOT head topology recorded in checkpoint args."""
    out_dim = _checkpoint_argument(checkpoint, "out_dim")
    patch_out_dim = _checkpoint_argument(checkpoint, "patch_out_dim", out_dim)
    shared_head = _checkpoint_argument(checkpoint, "shared_head_teacher")
    if shared_head is None:
        shared_head = _checkpoint_argument(checkpoint, "shared_head")
    if shared_head is None:
        shared_head = not any(
            name.startswith(("last_layer2.", "mlp2.", "last_norm2."))
            for name in head_state
        )
    norm = _checkpoint_argument(checkpoint, "norm_in_head")
    act = _checkpoint_argument(checkpoint, "act_in_head", "gelu")
    norm_last_layer = _checkpoint_argument(checkpoint, "norm_last_layer", True)

    if out_dim is None:
        output_weights = [
            value
            for name, value in head_state.items()
            if name in {"last_layer.weight_g", "last_layer.weight_v"}
        ]
        if not output_weights:
            raise ValueError("Cannot infer teacher head out_dim from checkpoint")
        out_dim = int(output_weights[0].shape[0])
    if patch_out_dim is None:
        patch_weights = [
            value
            for name, value in head_state.items()
            if name in {"last_layer2.weight_g", "last_layer2.weight_v"}
        ]
        patch_out_dim = int(patch_weights[0].shape[0]) if patch_weights else out_dim

    head = iBOTHead(
        backbone.embed_dim,
        int(out_dim),
        patch_out_dim=int(patch_out_dim),
        norm=norm,
        act=act,
        norm_last_layer=bool(norm_last_layer),
        shared_head=bool(shared_head),
    )
    head_state = _copy_shared_head_aliases(head_state, bool(shared_head))
    incompatible = head.load_state_dict(head_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Teacher projection head is incompatible with the checkpoint: "
            f"missing={incompatible.missing_keys}; "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return head, int(patch_out_dim if not shared_head else out_dim)


def _load_teacher(checkpoint_path: Path) -> TeacherModel:
    """Reuse load_backbone for the EMA backbone and restore its teacher head."""
    backbone, metadata = load_backbone(checkpoint_path, CHECKPOINT_KEY, "auto")
    checkpoint = _torch_load(checkpoint_path)
    raw_state = _teacher_state(checkpoint)
    head, concepts = _build_teacher_head(
        backbone, checkpoint, _teacher_head_state(raw_state)
    )

    mode = _checkpoint_argument(checkpoint, "teacher_target_ibot", "centering")
    if mode not in ("centering", "sinkhorn_knopp"):
        raise ValueError(f"Unsupported teacher_target_ibot mode: {mode!r}")
    temperature = float(
        _checkpoint_argument(checkpoint, "teacher_patch_temp", 0.07)
    )
    center = None
    ibot_loss = checkpoint.get("ibot_loss") if isinstance(checkpoint, Mapping) else None
    if mode == "centering":
        if not isinstance(ibot_loss, Mapping) or "center2" not in ibot_loss:
            raise ValueError(
                "Centering teacher patch probabilities requires ibot_loss.center2"
            )
        center = ibot_loss["center2"].detach().float()
        if center.ndim == 0 or center.shape[-1] != concepts or center.numel() != concepts:
            raise ValueError(
                "Checkpoint center2 dimension does not match patch head: "
                f"center2={tuple(center.shape)}, concepts={concepts}"
            )

    backbone = backbone.to(DEVICE).eval()
    head = head.to(DEVICE).eval()
    return TeacherModel(
        backbone=backbone,
        head=head,
        metadata=metadata,
        patch_target_mode=mode,
        patch_temperature=temperature,
        patch_center=center.to(DEVICE) if center is not None else None,
        concepts=concepts,
    )


def _patch_grid(model: torch.nn.Module, image: Image.Image, num_patches: int) -> tuple[int, int]:
    patch_size = getattr(model.patch_embed, "patch_size", None)
    if isinstance(patch_size, tuple):
        patch_height, patch_width = map(int, patch_size)
    else:
        patch_height = patch_width = int(patch_size)
    grid_height = image.height // patch_height
    grid_width = image.width // patch_width
    if (
        image.height % patch_height
        or image.width % patch_width
        or grid_height * grid_width != num_patches
    ):
        raise ValueError(
            "Cannot map patch tokens to the input image: "
            f"tokens={num_patches}, grid={(grid_height, grid_width)}, "
            f"patch_size={(patch_height, patch_width)}"
        )
    return grid_height, grid_width


def _num_special_tokens(model: torch.nn.Module) -> int:
    """Return the number of prefix/register tokens before spatial patches."""
    register_tokens = getattr(model, "num_register_tokens", None)
    if register_tokens is None:
        register = getattr(model, "register_tokens", None)
        register_tokens = 0 if register is None else int(register.shape[1])
    return 1 + int(register_tokens)


@torch.inference_mode()
def _extract_result(model: TeacherModel, image: Image.Image) -> ImageResult:
    tensor = MODEL_TRANSFORM(image).unsqueeze(0).to(DEVICE, non_blocking=True)
    tokens = model.backbone(tensor, return_all_tokens=True)
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"Expected [1, tokens, embedding_dim], got {tuple(tokens.shape)}")

    special_tokens = _num_special_tokens(model.backbone)
    patch_tokens = tokens[:, special_tokens:]
    # [num_patches, embedding_dim]: only spatial patch tokens enter k-means.
    patch_features = patch_tokens.squeeze(0).float().cpu().numpy()
    grid_height, grid_width = _patch_grid(
        model.backbone, image, patch_features.shape[0]
    )

    # iBOTHead expects one CLS token followed by spatial patches. Registers, if
    # present, are deliberately excluded before obtaining [num_patches, K].
    head_input = torch.cat((tokens[:, :1], patch_tokens), dim=1)
    _, patch_logits = model.head(head_input)
    patch_logits = patch_logits.squeeze(0).float()
    if patch_logits.ndim != 2 or patch_logits.shape[0] != patch_features.shape[0]:
        raise ValueError(
            "Patch head output does not match spatial features: "
            f"features={tuple(patch_features.shape)}, "
            f"logits={tuple(patch_logits.shape)}"
        )

    kmeans = KMeans(
        n_clusters=KMEANS_CLUSTERS,
        n_init=KMEANS_N_INIT,
        random_state=SEED,
    )
    labels = kmeans.fit_predict(patch_features).astype(np.int64, copy=False)
    object_cluster = _select_object_cluster(labels, grid_height, grid_width)

    if model.patch_target_mode == "centering":
        # Training stores center2 as [1, 1, K], while visualization logits are
        # [patches, K]. Drop singleton axes to avoid creating [1, patches, K].
        center = model.patch_center.reshape(model.concepts)
        probabilities = F.softmax(
            (patch_logits.to(DEVICE) - center) / model.patch_temperature,
            dim=-1,
        ).float().cpu()
    else:
        # Visualization has one image, so SK is applied jointly to all of that
        # image's patches, using the same raw-logit operation as training.
        probabilities = sinkhorn_knopp(
            patch_logits.to(DEVICE), model.patch_temperature
        ).float().cpu()

    # [num_pseudo_concepts]: average the teacher probabilities over the object cluster.
    object_mask = torch.from_numpy(labels == object_cluster)
    if not object_mask.any():
        raise RuntimeError("The selected object cluster has no patches")
    mean_probability = probabilities[object_mask].mean(dim=0).numpy()
    return ImageResult(
        cluster_map=labels.reshape(grid_height, grid_width),
        object_cluster=object_cluster,
        probabilities=mean_probability,
    )


def _select_object_cluster(labels: np.ndarray, grid_height: int, grid_width: int) -> int:
    """Choose the cluster most concentrated toward the image center."""
    rows, columns = np.indices((grid_height, grid_width), dtype=np.float32)
    row_distance = (rows - (grid_height - 1) / 2) / max(grid_height - 1, 1)
    column_distance = (columns - (grid_width - 1) / 2) / max(grid_width - 1, 1)
    centrality = 1.0 - np.sqrt(row_distance**2 + column_distance**2)
    scores = np.full(KMEANS_CLUSTERS, -np.inf, dtype=np.float64)
    for cluster in range(KMEANS_CLUSTERS):
        selected = labels.reshape(grid_height, grid_width) == cluster
        if selected.any():
            scores[cluster] = float(centrality[selected].mean())
    # np.argmax returns the lower cluster id on an exact tie.
    return int(np.argmax(scores))


def _cluster_overlay(
    image: Image.Image, result: ImageResult
) -> np.ndarray:
    """Nearest-neighbor two-color cluster map blended with the original image."""
    object_color = np.array([226, 61, 54], dtype=np.float32)
    background_color = np.array([44, 103, 190], dtype=np.float32)
    cluster_rgb = np.where(
        result.cluster_map[..., None] == result.object_cluster,
        object_color,
        background_color,
    ).astype(np.uint8)
    cluster_image = Image.fromarray(cluster_rgb, mode="RGB").resize(
        image.size, resample=Image.Resampling.NEAREST
    )
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    return (0.52 * original + 0.48 * np.asarray(cluster_image)).clip(0, 255).astype(
        np.uint8
    )


def _plot_probability_axis(
    axis,
    probabilities: np.ndarray,
    title: str,
    y_limit: float,
) -> None:
    concepts = np.arange(probabilities.shape[0])
    axis.vlines(concepts, 0.0, probabilities, color="#536d8a", linewidth=0.55)
    top_count = min(TOP_K_CONCEPTS, probabilities.shape[0])
    top_indices = np.argsort(probabilities)[-top_count:][::-1]
    axis.scatter(
        top_indices,
        probabilities[top_indices],
        color="#c43c39",
        s=13,
        zorder=3,
    )
    for concept in top_indices:
        axis.annotate(
            str(int(concept)),
            (concept, probabilities[concept]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8f2523",
        )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Pseudo-concept index")
    axis.set_ylabel("Mean teacher probability")
    axis.set_xlim(-0.5, max(probabilities.shape[0] - 0.5, 0.5))
    axis.set_ylim(0.0, y_limit)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)


def _save_comparison(
    sample_number: int,
    record: dict,
    original: Image.Image,
    results: list[tuple[str, ImageResult]],
) -> Path:
    y_limit = max(
        1e-6,
        max(float(result.probabilities.max()) for _, result in results) * 1.15,
    )
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.15, 1.15),
        width_ratios=(1.0, 1.0, 1.65),
    )
    original_axis = figure.add_subplot(grid[0, :])
    original_axis.imshow(original)
    original_axis.set_title(
        f"Original ImageNet image | {record['class_name']} | "
        f"dataset index {record['dataset_index']}",
        fontsize=11,
    )
    original_axis.axis("off")

    for row, (name, result) in enumerate(results, start=1):
        map_axis = figure.add_subplot(grid[row, :2])
        map_axis.imshow(_cluster_overlay(original, result))
        map_axis.set_title(
            f"{name}: two-cluster patch map "
            f"(object cluster {result.object_cluster})",
            fontsize=10,
        )
        map_axis.axis("off")
        probability_axis = figure.add_subplot(grid[row, 2])
        _plot_probability_axis(
            probability_axis,
            result.probabilities,
            f"{name}: patch concepts (K={result.probabilities.shape[0]})",
            y_limit,
        )

    figure.suptitle("Teacher patch concept comparison", fontsize=13)
    safe_class = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in record["class_name"]
    )
    path = Path(OUTPUT_DIR) / f"{sample_number:04d}_{safe_class}.png"
    figure.savefig(path, dpi=IMAGE_DPI, facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    _check_inputs()
    _set_deterministic_seed()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    dataset, indices = _sample_images()
    images, records = _prepare_selected_images(dataset, indices)

    checkpoints = [
        (CHECKPOINT_1_NAME, Path(CHECKPOINT_1)),
        (CHECKPOINT_2_NAME, Path(CHECKPOINT_2)),
    ]
    all_results: dict[str, list[ImageResult | None]] = {
        name: [None] * len(images) for name, _ in checkpoints
    }
    failures: list[dict] = []

    for name, checkpoint in checkpoints:
        print(f"Loading {name}: {checkpoint}", flush=True)
        model = _load_teacher(checkpoint)
        print(
            f"Loaded {name}: architecture={model.metadata['architecture']}, "
            f"patch_size={model.metadata['patch_size']}, "
            f"target_mode={model.patch_target_mode}, K={model.concepts}, "
            f"device={DEVICE}",
            flush=True,
        )
        try:
            for sample_number, image in enumerate(images):
                try:
                    all_results[name][sample_number] = _extract_result(model, image)
                    result = all_results[name][sample_number]
                    print(
                        f"[{name}] {sample_number + 1}/{len(images)} "
                        f"{records[sample_number]['class_name']} | "
                        f"patches={result.cluster_map.size} | "
                        f"K={result.probabilities.shape[0]}",
                        flush=True,
                    )
                except Exception as error:  # keep processing independent samples
                    failures.append(
                        {
                            "sample_number": sample_number,
                            "checkpoint": str(checkpoint),
                            "error": repr(error),
                        }
                    )
                    print(
                        f"[{name}] FAILED sample {sample_number}: {error}", flush=True
                    )
        finally:
            model.backbone.cpu()
            model.head.cpu()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    processed = 0
    for sample_number, (image, record) in enumerate(zip(images, records)):
        results = [
            (name, all_results[name][sample_number])
            for name, _ in checkpoints
            if all_results[name][sample_number] is not None
        ]
        if len(results) != len(checkpoints):
            continue
        _save_comparison(
            sample_number,
            record,
            image,
            [(name, result) for name, result in results if result is not None],
        )
        processed += 1

    print("\nDone.")
    print(f"Images processed: {processed}/{len(images)}")
    print(f"Checkpoint 1: {CHECKPOINT_1_NAME} -> {CHECKPOINT_1}")
    print(f"Checkpoint 2: {CHECKPOINT_2_NAME} -> {CHECKPOINT_2}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Failed images: {failures if failures else 'none'}")


if __name__ == "__main__":
    main()
