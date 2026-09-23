#!/usr/bin/env python3
"""COAT-style Object Compositionality on ORIDa (evaluation only)."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from evaluation.utils.common import load_backbone
from patch_concept_visualization import (
    _build_teacher_head, _checkpoint_argument, _num_special_tokens,
    _teacher_head_state, _teacher_state, _torch_load,
)


CHECKPOINTS = {
    "iBOT baseline": Path("checkpoints/ibot_vit_small.pth"),
    "Region +200": Path("checkpoints/checkpoint_source1000_continuation0200.pth"),
}
ORIDA_ROOT = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/datasets/orida/ORIDa_v1.0")
OUTPUT_DIR = Path("/mnt/fast/nobackup/scratch4weeks/jg02228/coat_orida")
DEVICE = "cuda"
INPUT_SIZE = 224
BATCH_SIZE = 12
MAX_SCENE_PAIRS_PER_OBJECT = 10
NUM_RANDOM_D = 64
CENTER_THRESHOLD = .20
MIN_AREA_RATIO = .5
MAX_AREA_RATIO = 2.0
SCALE_COST_WEIGHT = .25
EPS = 1e-12
ALPHA = .005
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = .95
SEED = 0
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)


@dataclass(frozen=True)
class FactualImage:
    image_path: str
    bbox: tuple[float, float, float, float]  # x, y, width, height
    position_index: str
    mask_path: str | None = None


@dataclass(frozen=True)
class FCFSet:
    object_id: str
    scene_id: str
    background_path: str
    factual_images: tuple[FactualImage, ...]


@dataclass(frozen=True)
class CoatTuple:
    tuple_id: str
    object_id: str
    scene_1_id: str
    scene_2_id: str
    A_path: str
    B_path: str
    C_path: str
    D_path: str
    B_position_index: str
    D_position_index: str
    B_cx: float
    B_cy: float
    D_cx: float
    D_cy: float
    B_area: float
    D_area: float
    center_distance: float
    scale_distance: float
    matching_cost: float
    D_alternative_paths: tuple[str, ...]


OBJECT_KEYS = ("object_id", "physical_object_id", "instance_id", "object_uid")
SCENE_KEYS = ("scene_id", "fcf_set_id", "set_id", "group_id")
BACKGROUND_KEYS = ("background_path", "background", "counterfactual_path", "empty_image")
FACTUALS_KEYS = ("factual_images", "factuals", "images_with_object", "positive_images")
IMAGE_KEYS = ("image_path", "path", "file_name", "filename", "image")
POSITION_KEYS = ("position_index", "position", "placement_index", "view_index")
MASK_KEYS = ("mask_path", "segmentation_path", "mask")
ROLE_KEYS = ("role", "type", "image_type", "kind")


def _first(record, names, default=None):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return default


def _resolve_path(value, root, metadata_parent):
    path = Path(str(value)).expanduser()
    candidates = [path] if path.is_absolute() else [metadata_parent / path, root / path]
    matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not matches:
        raise FileNotFoundError(f"ORIDa metadata references missing file: {value}")
    return str(matches[0])


def _parse_bbox(record):
    value = _first(record, ("bbox", "bounding_box", "box"))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [float(item) for item in value.replace(",", " ").split()]
    if isinstance(value, dict):
        value = [
            _first(value, ("x", "left", "xmin")),
            _first(value, ("y", "top", "ymin")),
            _first(value, ("width", "w")),
            _first(value, ("height", "h")),
        ]
    if value is None:
        x, y = _first(record, ("x", "left", "xmin")), _first(record, ("y", "top", "ymin"))
        width, height = _first(record, ("width", "bbox_width", "w")), _first(record, ("height", "bbox_height", "h"))
        value = [x, y, width, height]
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(item is None for item in value):
        raise ValueError("Each factual image requires a bbox in x,y,width,height form")
    bbox = tuple(float(item) for item in value)
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"Invalid non-positive bbox: {bbox}")
    return bbox


def _metadata_diagnostic(root, candidates):
    print("Could not construct the required ORIDa F-CF index.")
    print(f"Dataset root: {root}")
    print("Relevant dataset tree:")
    for path in sorted(root.rglob("*"))[:200]:
        print(" ", path.relative_to(root))
    print("Metadata fields discovered:")
    for path, payload in candidates:
        if isinstance(payload, dict):
            fields = sorted(payload)[:80]
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            fields = sorted(payload[0])[:80]
        else:
            fields = [type(payload).__name__]
        print(f"  {path}: {fields}")


def _load_metadata_candidates(root):
    output = []
    for path in sorted(root.rglob("*")):
        try:
            if path.suffix.casefold() == ".json":
                output.append((path, json.loads(path.read_text())))
            elif path.suffix.casefold() in {".csv", ".tsv"}:
                delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
                with path.open(newline="") as stream:
                    output.append((path, list(csv.DictReader(stream, delimiter=delimiter))))
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
            continue
    return output


def _nested_records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("fcf_sets", "f_cf_sets", "sets", "groups", "data", "annotations"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _adapt_nested(records, root, metadata_path):
    sets = []
    for record in records:
        if not isinstance(record, dict):
            return []
        object_id, scene_id = _first(record, OBJECT_KEYS), _first(record, SCENE_KEYS)
        background, factuals = _first(record, BACKGROUND_KEYS), _first(record, FACTUALS_KEYS)
        if object_id is None or scene_id is None or background is None or not isinstance(factuals, list):
            return []
        parsed = []
        for index, factual in enumerate(factuals):
            if not isinstance(factual, dict) or _first(factual, IMAGE_KEYS) is None:
                return []
            mask = _first(factual, MASK_KEYS)
            mask = mask if isinstance(mask, (str, Path)) else None
            parsed.append(FactualImage(
                _resolve_path(_first(factual, IMAGE_KEYS), root, metadata_path.parent),
                _parse_bbox(factual), str(_first(factual, POSITION_KEYS, index)),
                _resolve_path(mask, root, metadata_path.parent) if mask else None,
            ))
        if len(parsed) == 4:
            sets.append(FCFSet(
                str(object_id), str(scene_id),
                _resolve_path(background, root, metadata_path.parent), tuple(parsed),
            ))
    return sets


def _adapt_rows(records, root, metadata_path):
    if not records or not all(isinstance(record, dict) for record in records):
        return []
    groups = defaultdict(lambda: {"background": None, "factuals": []})
    for record in records:
        object_id, scene_id = _first(record, OBJECT_KEYS), _first(record, SCENE_KEYS)
        image_value = _first(record, IMAGE_KEYS)
        if object_id is None or scene_id is None or image_value is None:
            return []
        role = str(_first(record, ROLE_KEYS, "")).casefold()
        is_background = role in {"background", "counterfactual", "empty", "background-only", "cf"}
        key = (str(object_id), str(scene_id))
        path = _resolve_path(image_value, root, metadata_path.parent)
        declared_background = _first(record, BACKGROUND_KEYS)
        if declared_background is not None:
            groups[key]["background"] = _resolve_path(
                declared_background, root, metadata_path.parent
            )
        if is_background:
            groups[key]["background"] = path
        else:
            mask = _first(record, MASK_KEYS)
            mask = mask if isinstance(mask, (str, Path)) else None
            groups[key]["factuals"].append(FactualImage(
                path, _parse_bbox(record),
                str(_first(record, POSITION_KEYS, len(groups[key]["factuals"]))),
                _resolve_path(mask, root, metadata_path.parent) if mask else None,
            ))
    return [
        FCFSet(object_id, scene_id, item["background"], tuple(item["factuals"]))
        for (object_id, scene_id), item in groups.items()
        if item["background"] is not None and len(item["factuals"]) == 4
    ]


def _parse_orida_bbox(path, image_path):
    """Read ORIDa's normalized x1,y1,x2,y2 bbox into absolute xywh."""
    values = [float(value) for value in path.read_text().replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(f"Expected four x1,y1,x2,y2 values in {path}; got {values}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite ORIDa bbox values in {path}: {values}")
    x1, y1, x2, y2 = values
    with Image.open(image_path) as image:
        width, height = image.size
    if all(0.0 <= value <= 1.0 for value in values):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ORIDa bbox is outside image bounds in {path}: {values}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ORIDa x1,y1,x2,y2 box in {path}: {values}")
    return (x1, y1, x2 - x1, y2 - y1)


def _load_directory_fcf_sets(root):
    """Read the official ORIDa_v1.0 split/object/scene directory structure."""
    sets = []
    found_split = False
    for split_name in ("train", "validation"):
        split_root = root / split_name
        if not split_root.is_dir():
            continue
        found_split = True
        for object_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
            fcf_root = object_root / "factual_counterfactual"
            if not fcf_root.is_dir():
                continue
            for scene_root in sorted(path for path in fcf_root.iterdir() if path.is_dir()):
                images_root = scene_root / "images"
                if not images_root.is_dir():
                    continue
                images = {}
                for image_path in images_root.iterdir():
                    if not image_path.is_file() or image_path.suffix.casefold() not in {".jpg", ".jpeg"}:
                        continue
                    try:
                        position = int(image_path.stem.rsplit("_", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    if position in range(5):
                        if position in images:
                            raise ValueError(f"Duplicate ORIDa position {position}: {scene_root}")
                        images[position] = image_path
                if set(images) != set(range(5)):
                    raise ValueError(
                        f"Expected ORIDa images at positions 0..4 in {scene_root}; "
                        f"found {sorted(images)}"
                    )
                factuals = []
                for position in range(1, 5):
                    image_path = images[position]
                    bbox_path = (
                        scene_root / "annotations" / "bbox"
                        / f"{image_path.stem}_bbox.txt"
                    )
                    if not bbox_path.is_file():
                        raise FileNotFoundError(f"Missing ORIDa bbox annotation: {bbox_path}")
                    mask_path = (
                        scene_root / "annotations" / "masks"
                        / f"{image_path.stem}_mask.jpg"
                    )
                    factuals.append(FactualImage(
                        str(image_path.resolve()),
                        _parse_orida_bbox(bbox_path, image_path),
                        str(position),
                        str(mask_path.resolve()) if mask_path.is_file() else None,
                    ))
                sets.append(FCFSet(
                    str(object_root.name),
                    f"{split_name}:{scene_root.name}",
                    str(images[0].resolve()),
                    tuple(factuals),
                ))
    if not found_split:
        return []
    identities = [(item.object_id, item.scene_id) for item in sets]
    if len(identities) != len(set(identities)):
        raise ValueError("ORIDa folder tree contains duplicate object/scene F-CF sets")
    return sorted(sets, key=lambda item: (item.object_id, item.scene_id))


def load_orida_fcf_sets(root=ORIDA_ROOT):
    """Load official ORIDa folders, with JSON/CSV support for alternate exports."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Configure ORIDA_ROOT; directory not found: {root}")
    split_roots = [root / "train", root / "validation"]
    present_splits = [path.is_dir() for path in split_roots]
    if any(present_splits):
        if not all(present_splits):
            raise FileNotFoundError(
                "ORIDa extraction is incomplete; expected both train/ and validation/"
            )
        directory_sets = _load_directory_fcf_sets(root)
        if not directory_sets:
            raise ValueError("No factual-counterfactual ORIDa scenes were indexed")
        print(f"Indexed {len(directory_sets)} ORIDa factual-counterfactual scenes", flush=True)
        return directory_sets
    candidates = _load_metadata_candidates(root)
    valid = []
    errors = []
    for path, payload in candidates:
        records = _nested_records(payload)
        for adapter in (_adapt_nested, _adapt_rows):
            try:
                sets = adapter(records, root, path)
            except (ValueError, FileNotFoundError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            if sets:
                valid.append((path, sets))
    if not valid:
        _metadata_diagnostic(root, candidates)
        if errors:
            print("Adapter errors:", *errors[:20], sep="\n  ")
        raise ValueError("No usable ORIDa F-CF metadata source was found")
    valid.sort(key=lambda item: (-len(item[1]), str(item[0])))
    if len(valid) > 1 and len(valid[0][1]) == len(valid[1][1]):
        first_ids = {(item.object_id, item.scene_id) for item in valid[0][1]}
        second_ids = {(item.object_id, item.scene_id) for item in valid[1][1]}
        if first_ids != second_ids:
            _metadata_diagnostic(root, candidates)
            raise ValueError("Multiple incompatible ORIDa F-CF metadata sources were found")
    _, sets = valid[0]
    identities = [(item.object_id, item.scene_id) for item in sets]
    if len(identities) != len(set(identities)):
        raise ValueError("ORIDa metadata contains duplicate object_id/scene_id F-CF sets")
    return sorted(sets, key=lambda item: (item.object_id, item.scene_id))


def _geometry(factual):
    with Image.open(factual.image_path) as image:
        width, height = image.size
    x, y, box_width, box_height = factual.bbox
    return (
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width * box_height / (width * height),
    )


def build_tuples(fcf_sets, seed=SEED):
    from scipy.optimize import linear_sum_assignment

    by_object = defaultdict(list)
    for item in fcf_sets:
        if len(item.factual_images) == 4:
            by_object[item.object_id].append(item)
    rng = np.random.default_rng(seed)
    tuples = []
    for object_id in sorted(by_object):
        scene_pairs = list(itertools.combinations(sorted(by_object[object_id], key=lambda x: x.scene_id), 2))
        if len(scene_pairs) > MAX_SCENE_PAIRS_PER_OBJECT:
            selected = sorted(rng.choice(len(scene_pairs), MAX_SCENE_PAIRS_PER_OBJECT, replace=False))
            scene_pairs = [scene_pairs[index] for index in selected]
        for scene_1, scene_2 in scene_pairs:
            geometry_1 = [_geometry(item) for item in scene_1.factual_images]
            geometry_2 = [_geometry(item) for item in scene_2.factual_images]
            cost = np.empty((4, 4))
            details = {}
            for row, (cx_b, cy_b, area_b) in enumerate(geometry_1):
                for column, (cx_d, cy_d, area_d) in enumerate(geometry_2):
                    center = math.hypot(cx_b - cx_d, cy_b - cy_d)
                    scale = abs(math.log(area_b / area_d))
                    cost[row, column] = center + SCALE_COST_WEIGHT * scale
                    details[row, column] = (center, scale, area_b / area_d)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                center, scale, ratio = details[int(row), int(column)]
                if center > CENTER_THRESHOLD or not MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO:
                    continue
                b, d = scene_1.factual_images[int(row)], scene_2.factual_images[int(column)]
                cx_b, cy_b, area_b = geometry_1[int(row)]
                cx_d, cy_d, area_d = geometry_2[int(column)]
                tuple_id = f"{object_id}__{scene_1.scene_id}__{scene_2.scene_id}__{b.position_index}__{d.position_index}"
                alternatives = tuple(item.image_path for index, item in enumerate(scene_2.factual_images) if index != int(column))
                tuples.append(CoatTuple(
                    tuple_id, object_id, scene_1.scene_id, scene_2.scene_id,
                    scene_1.background_path, b.image_path, scene_2.background_path, d.image_path,
                    b.position_index, d.position_index, cx_b, cy_b, cx_d, cy_d,
                    area_b, area_d, center, scale, float(cost[row, column]), alternatives,
                ))
    if not tuples:
        raise ValueError("No valid COAT tuples survived ORIDa position/scale matching")
    return sorted(tuples, key=lambda item: item.tuple_id)


def build_random_baselines(tuples, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for positive in tuples:
        eligible = [
            candidate for candidate in tuples
            if candidate.tuple_id != positive.tuple_id
            and candidate.scene_2_id != positive.scene_2_id
            and candidate.D_path != positive.D_path
        ]
        eligible = list({item.D_path: item for item in eligible}.values())
        preferred = [item for item in eligible if item.object_id != positive.object_id]
        pool = preferred if len(preferred) >= NUM_RANDOM_D else eligible
        if len(pool) < NUM_RANDOM_D:
            raise ValueError(
                f"Tuple {positive.tuple_id} has only {len(pool)} random-D candidates; need {NUM_RANDOM_D}"
            )
        indices = rng.choice(len(pool), NUM_RANDOM_D, replace=False)
        for rank, index in enumerate(indices):
            candidate = pool[int(index)]
            rows.append({
                "tuple_id": positive.tuple_id, "random_rank": rank,
                "candidate_tuple_id": candidate.tuple_id,
                "candidate_object_id": candidate.object_id,
                "Dhat_path": candidate.D_path,
            })
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoints(checkpoints=CHECKPOINTS):
    if list(checkpoints) != ["iBOT baseline", "Region +200"]:
        raise ValueError("Configure exactly the iBOT baseline and Region +200 checkpoints")
    records = {}
    for label, configured_path in checkpoints.items():
        path = Path(configured_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = _torch_load(path)
        records[label] = {
            "path": str(path), "sha256": sha256(path),
            "lambda3": _checkpoint_argument(checkpoint, "lambda3", None),
            "arch": _checkpoint_argument(checkpoint, "arch", None),
            "patch_size": _checkpoint_argument(checkpoint, "patch_size", None),
            "prototype_dim": _checkpoint_argument(checkpoint, "patch_out_dim", None),
            "region_normalization": _checkpoint_argument(checkpoint, "region_normalization", None),
            "region_temp": _checkpoint_argument(checkpoint, "region_temp", None),
        }
    if records["iBOT baseline"]["sha256"] == records["Region +200"]["sha256"]:
        raise ValueError("iBOT baseline and region-trained checkpoints are byte-identical")
    region_lambda = records["Region +200"]["lambda3"]
    if region_lambda is not None and float(region_lambda) <= 0:
        raise ValueError(f"Region checkpoint lambda3 must be positive; got {region_lambda}")
    for key in ("arch", "patch_size", "prototype_dim"):
        values = [records[label][key] for label in checkpoints]
        if all(value is not None for value in values) and values[0] != values[1]:
            raise ValueError(f"Checkpoint {key} mismatch: {values}")
    architecture = records["Region +200"]["arch"]
    if architecture is not None and "small" not in str(architecture).casefold():
        raise ValueError(f"Expected ViT-S/16 checkpoint architecture; got {architecture!r}")
    normalization = records["Region +200"]["region_normalization"]
    if normalization not in (None, "centering", "softmax", "raw_logits", "sinkhorn"):
        raise ValueError(f"Unknown training region normalization: {normalization!r}")
    temperature = records["Region +200"]["region_temp"]
    temperature = .1 if temperature is None else float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid region temperature: {temperature}")
    return records, temperature


def load_teacher(path):
    backbone, metadata = load_backbone(path, "teacher", "auto")
    checkpoint = _torch_load(path)
    head, prototypes = _build_teacher_head(
        backbone, checkpoint, _teacher_head_state(_teacher_state(checkpoint))
    )
    backbone = backbone.to(DEVICE).eval().requires_grad_(False)
    head = head.to(DEVICE).eval().requires_grad_(False)
    return backbone, head, metadata, prototypes


def validate_loaded_checkpoint_shapes(records):
    """Validate actual teacher/head shapes before indexing or evaluating data."""
    shapes = set()
    for label in CHECKPOINTS:
        backbone, head, metadata, prototypes = load_teacher(records[label]["path"])
        patch_size = int(metadata["patch_size"])
        records[label].update({
            "loaded_patch_size": patch_size,
            "loaded_prototype_dim": int(prototypes),
        })
        shapes.add((patch_size, int(prototypes)))
        del backbone, head
    if str(DEVICE).startswith("cuda"):
        torch.cuda.empty_cache()
    if len(shapes) != 1:
        raise ValueError(f"Loaded teacher patch-head dimensions do not match: {shapes}")
    patch_size, _ = next(iter(shapes))
    if patch_size != 16:
        raise ValueError(f"Expected ViT-S/16 patch size 16; got {patch_size}")


def load_rgb_tensor(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
    return TF.to_tensor(image)


@torch.inference_mode()
def encode_tensor_batch(raw_rgb, backbone, head, patch_size, prototypes, temperature):
    normalized = TF.normalize(raw_rgb, IMAGENET_MEAN, IMAGENET_STD).to(DEVICE)
    tokens = backbone(normalized, return_all_tokens=True)
    special = _num_special_tokens(backbone)
    spatial = tokens[:, special:]
    expected = (INPUT_SIZE // patch_size) ** 2
    if spatial.shape[1] != expected:
        raise ValueError(f"Expected {expected} spatial tokens after excluding {special} special tokens; got {spatial.shape[1]}")
    _, logits = head(torch.cat((tokens[:, :1], spatial), dim=1))
    if tuple(logits.shape) != (len(raw_rgb), expected, prototypes):
        raise ValueError(f"Unexpected patch-head shape: {tuple(logits.shape)}")
    representations = (logits.float() / temperature).softmax(-1).mean(1)
    representations /= representations.sum(1, keepdim=True)
    if representations.ndim != 2 or torch.any(representations < 0) or not torch.allclose(
        representations.sum(1),
        torch.ones(len(raw_rgb), device=representations.device),
        atol=1e-5,
    ):
        raise ValueError("Invalid patch-distribution representation")
    return representations.cpu().numpy()


def encode_paths(paths, backbone, head, patch_size, prototypes, temperature):
    paths = sorted(set(paths))
    cache = {}
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        raw = torch.stack([load_rgb_tensor(path) for path in batch_paths])
        encoded = encode_tensor_batch(raw, backbone, head, patch_size, prototypes, temperature)
        cache.update(zip(batch_paths, encoded))
    return cache


def l2_loss(a, b, c, d):
    residual = np.asarray(b) - np.asarray(a) + np.asarray(c) - np.asarray(d)
    return float(residual @ residual)


def angular_loss(a, b, c, d, eps=EPS):
    delta_ab, delta_cd = np.asarray(b) - np.asarray(a), np.asarray(d) - np.asarray(c)
    denominator = float(np.linalg.norm(delta_ab) * np.linalg.norm(delta_cd))
    if denominator <= eps:
        return None
    cosine = float(np.clip(delta_ab @ delta_cd / denominator, -1, 1))
    return float(np.arccos(cosine))


def coat_score(positive, baseline, eps=EPS):
    return 1.0 - float(positive) / (float(baseline) + eps)


def _mean_valid(values):
    valid = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def evaluate_model(label, checkpoint_record, tuples, random_rows, temperature):
    backbone, head, metadata, prototypes = load_teacher(checkpoint_record["path"])
    patch_size = int(metadata["patch_size"])
    if patch_size != 16 or INPUT_SIZE // patch_size != 14:
        raise ValueError(f"Expected ViT-S/16 14x14 grid; got patch_size={patch_size}")
    real_paths = set()
    for item in tuples:
        real_paths.update((item.A_path, item.B_path, item.C_path, item.D_path, *item.D_alternative_paths))
    real_paths.update(row["Dhat_path"] for row in random_rows)
    representations = encode_paths(real_paths, backbone, head, patch_size, prototypes, temperature)

    pixel_inputs = []
    for item in tuples:
        a, b, c = (load_rgb_tensor(path) for path in (item.A_path, item.B_path, item.C_path))
        pixel = b - a + c
        if float(torch.sum((b - a + c - pixel) ** 2)) >= 1e-8:
            raise AssertionError("Raw pixel-algebra sanity check failed")
        pixel_inputs.append(pixel)
    pixel_representations = []
    for start in range(0, len(pixel_inputs), BATCH_SIZE):
        batch = torch.stack(pixel_inputs[start:start + BATCH_SIZE])
        pixel_representations.extend(
            encode_tensor_batch(batch, backbone, head, patch_size, prototypes, temperature)
        )

    random_by_tuple = defaultdict(list)
    for row in random_rows:
        random_by_tuple[row["tuple_id"]].append(row["Dhat_path"])
    measurements = []
    angular_excluded = 0
    for index, item in enumerate(tuples):
        a, b, c, d = (representations[path] for path in (item.A_path, item.B_path, item.C_path, item.D_path))
        positive_l2, positive_acos = l2_loss(a, b, c, d), angular_loss(a, b, c, d)
        random_l2 = [l2_loss(a, b, c, representations[path]) for path in random_by_tuple[item.tuple_id]]
        random_acos = [angular_loss(a, b, c, representations[path]) for path in random_by_tuple[item.tuple_id]]
        baseline_l2, baseline_acos = float(np.mean(random_l2)), _mean_valid(random_acos)
        drop_l2, drop_acos = l2_loss(a, b, c, c), angular_loss(a, b, c, c)
        position_l2 = [l2_loss(a, b, c, representations[path]) for path in item.D_alternative_paths]
        position_acos = [angular_loss(a, b, c, representations[path]) for path in item.D_alternative_paths]
        position_acos_valid = [value for value in position_acos if value is not None]
        hard_position_l2 = min(position_l2)
        hard_position_acos = min(position_acos_valid) if position_acos_valid else None
        pixel_d = pixel_representations[index]
        pixel_l2, pixel_acos = l2_loss(a, b, c, pixel_d), angular_loss(a, b, c, pixel_d)
        if positive_acos is None:
            angular_excluded += 1
        measurements.append({
            "model": label, "tuple_id": item.tuple_id, "object_id": item.object_id,
            "l2_positive": positive_l2, "l2_random_baseline": baseline_l2,
            "coat_l2": coat_score(positive_l2, baseline_l2),
            "acos_positive": positive_acos, "acos_random_baseline": baseline_acos,
            "coat_acos": coat_score(positive_acos, baseline_acos) if positive_acos is not None and baseline_acos is not None else None,
            "l2_drop": drop_l2, "acos_drop": drop_acos,
            "l2_position_hard": hard_position_l2, "acos_position_hard": hard_position_acos,
            "l2_pixel": pixel_l2, "acos_pixel": pixel_acos,
            "pass_drop_l2": positive_l2 < drop_l2,
            "pass_drop_acos": positive_acos is not None and drop_acos is not None and positive_acos < drop_acos,
            "pass_position_l2": positive_l2 < hard_position_l2,
            "pass_position_acos": positive_acos is not None and hard_position_acos is not None and positive_acos < hard_position_acos,
            "pass_pixel_l2": positive_l2 < pixel_l2,
            "pass_pixel_acos": positive_acos is not None and pixel_acos is not None and positive_acos < pixel_acos,
        })
    del backbone, head
    if str(DEVICE).startswith("cuda"):
        torch.cuda.empty_cache()
    checkpoint_record.update({"loaded_patch_size": patch_size, "loaded_prototype_dim": int(prototypes)})
    return measurements, angular_excluded


def _object_means(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        value = row[metric]
        if value is not None and math.isfinite(float(value)):
            grouped[row["object_id"]].append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def hard_negative_statistics(rows, metric, negative):
    from scipy.stats import binomtest

    key = f"pass_{negative}_{metric}"
    negative_key = f"{metric}_{'position_hard' if negative == 'position' else negative}"
    valid = [
        bool(row[key]) for row in rows
        if row[f"{metric}_positive"] is not None and row[negative_key] is not None
    ]
    successes, count = sum(valid), len(valid)
    pvalue = float(binomtest(successes, count, p=.5, alternative="greater").pvalue) if count else 1.0
    return successes / count if count else math.nan, pvalue, count


def paired_bootstrap(all_rows):
    labels = list(CHECKPOINTS)
    rng = np.random.default_rng(SEED)
    output, distributions = {}, {}
    for metric in ("coat_l2", "coat_acos"):
        means = {label: _object_means(all_rows[label], metric) for label in labels}
        objects = sorted(set(means[labels[0]]) & set(means[labels[1]]))
        if len(objects) < 2:
            raise ValueError(f"Need at least two common objects for paired {metric} bootstrap")
        draws = {label: np.empty(BOOTSTRAP_SAMPLES) for label in labels}
        difference = np.empty(BOOTSTRAP_SAMPLES)
        for index in range(BOOTSTRAP_SAMPLES):
            sampled = rng.choice(objects, len(objects), replace=True)
            for label in labels:
                draws[label][index] = np.mean([means[label][item] for item in sampled])
            difference[index] = draws[labels[1]][index] - draws[labels[0]][index]
        alpha = (1 - CONFIDENCE) / 2
        output[metric] = {
            "objects": len(objects),
            labels[0]: (float(np.mean([means[labels[0]][item] for item in objects])), *map(float, np.quantile(draws[labels[0]], (alpha, 1-alpha)))),
            labels[1]: (float(np.mean([means[labels[1]][item] for item in objects])), *map(float, np.quantile(draws[labels[1]], (alpha, 1-alpha)))),
            "difference": (
                float(np.mean([means[labels[1]][item] - means[labels[0]][item] for item in objects])),
                *map(float, np.quantile(difference, (alpha, 1-alpha))),
            ),
        }
        distributions[metric] = difference
    return output


def summarize(all_rows, bootstrap):
    summaries = []
    for label, rows in all_rows.items():
        record = {"model": label, "objects": len({row['object_id'] for row in rows}), "tuples": len(rows)}
        for metric in ("l2", "acos"):
            coat_key = f"coat_{metric}"
            object_means = _object_means(rows, coat_key)
            values = [row[coat_key] for row in rows if row[coat_key] is not None]
            point, low, high = bootstrap[coat_key][label]
            record.update({
                f"coat_{metric}_macro": point, f"coat_{metric}_ci_low": low,
                f"coat_{metric}_ci_high": high, f"coat_{metric}_micro": float(np.mean(values)),
            })
            pvalues = []
            for negative in ("drop", "position", "pixel"):
                rate, pvalue, count = hard_negative_statistics(rows, metric, negative)
                record[f"{negative}_pass_rate_{metric}"] = rate
                record[f"{negative}_pvalue_{metric}"] = pvalue
                record[f"{negative}_valid_tuples_{metric}"] = count
                pvalues.append(pvalue)
            record[f"valid_{metric}"] = all(value < ALPHA for value in pvalues)
        summaries.append(record)
    return summaries


def _write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tuple_row(item):
    row = asdict(item)
    row["D_alternative_paths"] = json.dumps(row["D_alternative_paths"])
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_records, temperature = validate_checkpoints()
    validate_loaded_checkpoint_shapes(checkpoint_records)
    fcf_sets = load_orida_fcf_sets()
    index_rows = []
    for item in fcf_sets:
        for factual in item.factual_images:
            index_rows.append({
                "object_id": item.object_id, "scene_id": item.scene_id,
                "background_path": item.background_path, "image_path": factual.image_path,
                "bbox": json.dumps(factual.bbox), "mask_path": factual.mask_path,
                "position_index": factual.position_index,
            })
    _write_csv(OUTPUT_DIR / "orida_index.csv", index_rows)
    tuples = build_tuples(fcf_sets)
    _write_csv(OUTPUT_DIR / "coat_tuples.csv", [_tuple_row(item) for item in tuples])
    random_rows = build_random_baselines(tuples)
    _write_csv(OUTPUT_DIR / "random_baselines.csv", random_rows)

    all_rows, angular_excluded = {}, {}
    for label in CHECKPOINTS:
        print(f"Evaluating {label}", flush=True)
        rows, excluded = evaluate_model(label, checkpoint_records[label], tuples, random_rows, temperature)
        all_rows[label], angular_excluded[label] = rows, excluded
    measurements = [row for label in CHECKPOINTS for row in all_rows[label]]
    _write_csv(OUTPUT_DIR / "measurements.csv", measurements)
    bootstrap = paired_bootstrap(all_rows)
    summary = summarize(all_rows, bootstrap)
    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    comparison = [{
        "metric": metric, "region_minus_ibot_baseline": values["difference"][0],
        "ci_low": values["difference"][1], "ci_high": values["difference"][2],
        "objects": values["objects"],
    } for metric, values in bootstrap.items()]
    _write_csv(OUTPUT_DIR / "comparison.csv", comparison)

    protocol = {
        "name": "COAT-style Object Compositionality on ORIDa",
        "checkpoints": checkpoint_records, "evaluation_temperature": temperature,
        "input_resolution": INPUT_SIZE,
        "preprocessing": "full RGB scene; bicubic 224x224; ImageNet normalization; no augmentation",
        "representation": "mean of per-patch softmax(raw uncentered teacher patch-head logits / T)",
        "region_training_normalization": checkpoint_records["Region +200"]["region_normalization"],
        "orida_root": str(Path(ORIDA_ROOT).expanduser().resolve()),
        "indexed_fcf_sets": len(fcf_sets), "objects": len({item.object_id for item in tuples}),
        "selected_scene_pairs": len({(item.object_id, item.scene_1_id, item.scene_2_id) for item in tuples}),
        "final_tuples": len(tuples), "center_threshold": CENTER_THRESHOLD,
        "area_ratio_range": [MIN_AREA_RATIO, MAX_AREA_RATIO], "scale_cost_weight": SCALE_COST_WEIGHT,
        "random_baseline_count": NUM_RANDOM_D, "seed": SEED,
        "hard_negatives": {
            "drop": "D=C", "position": "lowest loss among other three target-scene positions",
            "pixel": "unclamped float D=B-A+C before ImageNet normalization",
        },
        "hard_negative_alpha": ALPHA, "angular_tuples_excluded": angular_excluded,
        "bootstrap_unit": "physical object_id", "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "confidence": CONFIDENCE,
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    print("\nCOAT-style object algebra on ORIDa\n" + "-" * 34)
    print(f"Objects: {protocol['objects']}\nTuples: {len(tuples)}\n")
    print(f"{'':24} {'COAT-L2':>12} {'COAT-acos':>12}")
    for row in summary:
        print(f"{row['model']:24} {row['coat_l2_macro']:12.4f} {row['coat_acos_macro']:12.4f}")
    print(f"{'Region - iBOT baseline':24} {bootstrap['coat_l2']['difference'][0]:12.4f} {bootstrap['coat_acos']['difference'][0]:12.4f}")
    print("\n95% CI of difference:")
    for metric in ("coat_l2", "coat_acos"):
        value = bootstrap[metric]["difference"]
        print(f"{metric}: [{value[1]:.4f}, {value[2]:.4f}]")
    print("\nHard-negative tests (drop / position / pixel):")
    for row in summary:
        for metric in ("l2", "acos"):
            rates = [row[f"{negative}_pass_rate_{metric}"] for negative in ("drop", "position", "pixel")]
            validity = "PASS" if row[f"valid_{metric}"] else "FAIL"
            print(f"{row['model']} {metric}: " + " / ".join(f"{value:.3f}" for value in rates) + f"  {validity}")
            if not row[f"valid_{metric}"]:
                print("WARNING: COAT score should not be interpreted as evidence of compositionality because the representation failed the shortcut-detection test.")


if __name__ == "__main__":
    main()
