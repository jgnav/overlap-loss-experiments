"""Video propagation using DINOv3 Sec. 6.1.5 / App. D.5 / Table 27.

The paper takes precedence over the demonstration notebook for frame geometry
and native mask interpolation. Unspecified implementation details follow the
released notebook. Custom YouTube-VOS/MOSE splits must be provided explicitly;
official challenge validation sets are different benchmarks.
The requested default averages the last four normalized blocks; the paper's
single-block extraction remains available through --video-feature-blocks 1.
"""

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from evaluation.utils.common import (
    base_parser, checkpoint_fingerprint, evaluation_identity, load_backbone,
    prepare_paths, print_progress, utc_now, write_json,
)
from evaluation.vendor.davis.metrics import db_eval_boundary, db_eval_iou


PAPER = "https://arxiv.org/abs/2508.10104v1"
NOTEBOOK = ("https://github.com/facebookresearch/dinov3/blob/"
            "6876159a11b4df116f30f667f8c9888617df0751/notebooks/segmentation_tracking.ipynb")
RESOLUTIONS = {14: {"small": 420, "medium": 840, "large": 1260},
               16: {"small": 480, "medium": 960, "large": 1440}}
NORMALIZE = T.Normalize((.485, .456, .406), (.229, .224, .225))


def resized_size(native_wh, short_side, patch_size):
    """Round independently scaled axes, matching both examples in footnote 2."""
    width, height = native_wh
    if min(width, height, short_side, patch_size) <= 0:
        raise ValueError("Image dimensions, short side and patch size must be positive")
    scale = short_side / min(width, height)
    return tuple(max(patch_size, math.floor(side * scale / patch_size + .5) * patch_size)
                 for side in (width, height))


def read_image(path, short_side, patch_size):
    # pathlib.Path and zipfile.Path both provide a binary stream. This allows
    # all-frame YouTube-VOS archives without extracting gigabytes of RGB data.
    with path.open("rb") as stream, Image.open(stream) as source:
        image = source.convert("RGB")
        native_wh = image.size
        image = image.resize(resized_size(native_wh, short_side, patch_size), Image.Resampling.BICUBIC)
        return NORMALIZE(T.ToTensor()(image)), native_wh


@torch.no_grad()
def patch_features(backbone, image, patch_size, feature_blocks=4):
    # This repository's API returns normalized [CLS, patches], removing any
    # registers internally. Average normalized blocks before L2 normalization.
    if feature_blocks not in (1, 4):
        raise ValueError("Video features require 1 or 4 blocks")
    layers = backbone.get_intermediate_layers(image[None], n=feature_blocks)
    if len(layers) != feature_blocks:
        raise ValueError("Backbone did not return the requested feature blocks")
    tokens = torch.stack([layer[0, 1:].float() for layer in layers]).mean(0)
    grid = tuple(int(side // patch_size) for side in image.shape[-2:])
    if tokens.shape[0] != grid[0] * grid[1]:
        raise ValueError("Backbone patch tokens do not match the resized frame grid")
    return F.normalize(tokens, dim=-1), grid


def initial_probabilities(mask, object_ids, grid, device):
    labels = torch.as_tensor(mask.astype(np.int64), device=device)
    labels = F.interpolate(labels[None, None].float(), size=grid, mode="nearest-exact")[0, 0].long()
    # Other IDs (including void) receive background probability. They are never
    # introduced as new channels by later ground-truth annotations.
    foreground = torch.stack([(labels == value).float() for value in object_ids])
    return torch.cat((1 - foreground.sum(0, keepdim=True), foreground)).flatten(1)


def propagate(target, features, probabilities, topk=5, temperature=.2, chunk_size=128):
    """Released top-k propagation, chunked to avoid quadratic GPU storage.

    Table 27's highlighted setting has an infinite spatial neighborhood.
    Ranking includes all ties at the kth similarity, as in the notebook.
    """
    if not features or len(features) != len(probabilities) or topk < 1 or temperature <= 0:
        raise ValueError("Invalid propagation context or hyperparameters")
    source = torch.cat(features)
    masks = torch.cat(probabilities, dim=1)
    if source.shape[0] != masks.shape[1] or source.shape[1] != target.shape[1]:
        raise ValueError("Context features and probabilities do not align")
    output = torch.empty((masks.shape[0], target.shape[0]), device=target.device, dtype=torch.float32)
    for begin in range(0, len(target), chunk_size):
        similarity = target[begin:begin + chunk_size] @ source.T
        cutoff = similarity.topk(min(topk, source.shape[0]), dim=1).values[:, -1:]
        similarity.masked_fill_(similarity < cutoff, -torch.inf)
        weights = (similarity / temperature).softmax(1)
        current = masks @ weights.T
        output[:, begin:begin + len(current.T)] = current / current.sum(0, keepdim=True)
    return output


def prediction(probabilities, grid, native_wh, object_ids):
    native = F.interpolate(probabilities.reshape(1, -1, *grid),
                           size=(native_wh[1], native_wh[0]), mode="bilinear", align_corners=False)
    # Released postprocess_probs: constant channels become zero (not one).
    minimum = native.flatten(2).amin(2)[:, :, None, None]
    maximum = native.flatten(2).amax(2)[:, :, None, None]
    native = torch.nan_to_num((native - minimum) / (maximum - minimum), nan=0)
    indices = native[0].argmax(0).cpu().numpy()
    return np.asarray((0, *object_ids), dtype=np.uint8)[indices]


def _frames(folder):
    if isinstance(folder, zipfile.Path):
        archive = folder.root
        if not hasattr(archive, "video_frame_index"):
            index = {}
            for name in archive.namelist():
                if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    parent = name.rsplit("/", 1)[0] + "/"
                    index.setdefault(parent, []).append(name)
            archive.video_frame_index = index
        frames = [zipfile.Path(archive, name) for name in sorted(archive.video_frame_index.get(folder.at, []))]
    else:
        frames = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(frames) < 2:
        raise ValueError(f"Video must have at least two frames: {folder}")
    return frames


def evaluation_frames(folder, mask_folder, dataset):
    frames = _frames(folder)
    original_count = len(frames)
    start = 0
    if dataset == "youtube_vos" and not (mask_folder / f"{frames[0].stem}.png").is_file():
        # The all-frame release includes leading RGB frames before the first
        # supplied label in some clips. Define the benchmark clip at its first
        # annotated frame; never seed new objects after that initialization.
        start = next((i for i, frame in enumerate(frames)
                      if (mask_folder / f"{frame.stem}.png").is_file()), len(frames))
    frames = frames[start:]
    if len(frames) < 2:
        raise ValueError(f"No evaluable labeled clip: {mask_folder}")
    return frames, {"original_rgb_frames": original_count, "skipped_leading_unannotated_frames": start,
                    "initialization_frame": frames[0].stem}


def dataset_layout(root, dataset, manifest_path=None):
    root = Path(root)
    if dataset == "davis":
        folder = next((root / name for name in ("DAVIS", "DAVIS2017", "davis2017")
                       if (root / name).is_dir()), None)
        if folder is None:
            raise FileNotFoundError(f"DAVIS 2017 missing under {root}")
        split = folder / "ImageSets/2017/val.txt"
        names = split.read_text().split()
        if len(names) != 30 or len(set(names)) != 30:
            raise ValueError("DINOv3 requires the 30-video DAVIS 2017 validation split")
        return folder / "JPEGImages/480p", folder / "Annotations/480p", names, {
            "dataset": "DAVIS 2017 val", "split_list": str(split),
            "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(), "author_split_verified": True,
        }
    if manifest_path is None:
        raise FileNotFoundError(
            f"DINOv3 {dataset} requires an explicit custom split manifest; "
            "the released paper does not provide its random seed or video lists. "
            "Official YouTube-VOS validation and MOSEv2 are different benchmarks. "
            "See docs/video_evaluation.md.")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    release = manifest.get("dataset_release")
    allowed_releases = {"2018", "2019"} if dataset == "youtube_vos" else {"2023", "v2"}
    expected = (2758, 690) if dataset == "youtube_vos" else (1206, 301)
    if (manifest.get("dataset") != dataset or release not in allowed_releases
            or not manifest.get("source") or type(manifest.get("author_split_verified")) is not bool):
        raise ValueError(f"Explicit dataset release, split provenance and verification required: {manifest_path}")
    splits = manifest.get("splits", {})
    selection, held_out = splits.get("selection"), splits.get("evaluation")
    if not isinstance(selection, list) or not isinstance(held_out, list) or not held_out:
        raise ValueError("Split manifests require selection/evaluation ID lists and a nonempty evaluation set")
    if manifest["author_split_verified"] and ((len(selection), len(held_out)) != expected or release == "v2"):
        raise ValueError(f"DINOv3 {dataset} split sizes must be {expected}, found {(len(selection), len(held_out))}")
    if (any(not isinstance(name, str) or not name or name in {".", ".."} or "\\" in name or Path(name).name != name
            for name in [*selection, *held_out]) or len(set(selection)) != len(selection)
            or len(set(held_out)) != len(held_out) or set(selection).intersection(held_out)):
        raise ValueError("Invalid, duplicated or overlapping video split IDs")
    def resolved(key):
        value = Path(manifest[key]).expanduser()
        base = root if manifest.get("paths_relative_to") == "datasets_root" else manifest_path.parent
        return value.resolve() if value.is_absolute() else (base / value).resolve()
    image_root = resolved("image_root")
    archive_metadata = {}
    if manifest.get("rgb_archive"):
        archive_path = resolved("rgb_archive")
        archive = zipfile.ZipFile(archive_path)
        prefix = manifest.get("rgb_archive_prefix", "")
        if not prefix or prefix.startswith("/") or ".." in Path(prefix).parts:
            archive.close()
            raise ValueError("An archive requires a safe explicit RGB directory prefix")
        image_root = zipfile.Path(archive, prefix.rstrip("/") + "/")
        # CRC/file-size/name identify archive frame contents without repeatedly
        # reading a multi-GB archive. ZipFile verifies CRC during decompression.
        index = hashlib.sha256()
        for info in archive.infolist():
            index.update(f"{info.filename}:{info.CRC}:{info.file_size}\n".encode())
        archive_metadata = {"rgb_archive": str(archive_path), "rgb_archive_index_sha256": index.hexdigest(),
                            "frame_sampling": "all released RGB frames; score annotated frames only"}
    return image_root, resolved("mask_root"), held_out, {
        "dataset": manifest.get("label", f"{'YouTube-VOS' if dataset == 'youtube_vos' else 'MOSE'} {release} custom held-out"),
        "split_manifest": str(manifest_path.resolve()),
        "split_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "author_split_verified": manifest["author_split_verified"], "split_source": manifest["source"],
        "dataset_release": release, "split_name": manifest.get("split_name"),
        "frame_sampling": manifest.get("frame_sampling", "released RGB cadence"), **archive_metadata,
    }


def preflight_masks(root, dataset, manifest_path=None):
    images, masks, names, details = dataset_layout(root, dataset, manifest_path)
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Missing RGB/annotation roots: {images}, {masks}")
    for name in names:
        frames, _ = evaluation_frames(images / name, masks / name, dataset)
        first = masks / name / f"{frames[0].stem}.png"
        if not first.is_file():
            raise FileNotFoundError(f"First-frame annotation missing: {first}")
        if dataset == "youtube_vos":
            frame_ids = {frame.stem for frame in frames}
            annotations = sorted((masks / name).glob("*.png"))
            if len(annotations) < 2 or any(path.stem not in frame_ids for path in annotations):
                raise ValueError(f"Missing scoring annotations or RGB alignment: {masks / name}")
        else:
            for frame in frames:
                path = masks / name / f"{frame.stem}.png"
                if not path.is_file():
                    raise FileNotFoundError(f"DINOv3 offline scoring annotation missing: {path}")
    return images, masks, names, details


@torch.no_grad()
def score_video(backbone, patch_size, frames, mask_folder, short_side, device, dataset, feature_blocks=4):
    with Image.open(mask_folder / f"{frames[0].stem}.png") as source:
        first_mask = np.asarray(source).copy()
    ids = tuple(int(value) for value in np.unique(first_mask) if value not in (0, 255))
    if not ids:
        return {}, {"status": "excluded_no_first_frame_object", "frames": len(frames)}
    image, native_wh = read_image(frames[0], short_side, patch_size)
    if first_mask.shape != (native_wh[1], native_wh[0]):
        raise ValueError("Initial annotation and RGB frame dimensions differ")
    first_features, grid = patch_features(backbone, image.to(device), patch_size, feature_blocks)
    first_probs = initial_probabilities(first_mask, ids, grid, device)
    history = deque(maxlen=7)
    scores = {str(value): [] for value in ids}
    for index, frame in enumerate(frames[1:], start=1):
        image, current_wh = read_image(frame, short_side, patch_size)
        features, current_grid = patch_features(backbone, image.to(device), patch_size, feature_blocks)
        if current_wh != native_wh or current_grid != grid:
            raise ValueError("Video frame dimensions changed")
        references = [(first_features, first_probs), *history]
        probabilities = propagate(features, [pair[0] for pair in references], [pair[1] for pair in references])
        history.append((features, probabilities))
        if dataset == "davis" and index == len(frames) - 1:
            continue  # Standard DAVIS semi-supervised scorer excludes endpoints.
        annotation = mask_folder / f"{frame.stem}.png"
        if dataset == "youtube_vos" and not annotation.is_file():
            continue  # Predict every RGB frame; score only released GT frames.
        with Image.open(annotation) as source:
            truth = np.asarray(source).copy()
        if truth.shape != first_mask.shape:
            raise ValueError(f"Scoring annotation dimensions differ: {frame}")
        predicted = prediction(probabilities, grid, native_wh, ids)
        void = truth == 255
        for object_id in ids:
            # Later object IDs never introduce new mask channels or GT seeds.
            gt, estimate = truth == object_id, predicted == object_id
            scores[str(object_id)].append((float(db_eval_iou(gt, estimate, void)),
                                          float(db_eval_boundary(gt, estimate, void))))
    if any(not values for values in scores.values()):
        raise ValueError(f"Video has no scored frames: {mask_folder}")
    return {key: {"j": float(np.mean([v[0] for v in values])),
                  "f": float(np.mean([v[1] for v in values]))} for key, values in scores.items()}, {
        "status": "completed", "frames": len(frames), "first_frame_object_ids": list(ids),
        "scored_frames": len(next(iter(scores.values()))), "native_size_wh": list(native_wh),
        "input_size_wh": list(resized_size(native_wh, short_side, patch_size)), "feature_grid_hw": list(grid),
    }


def main(dataset):
    parser = base_parser(f"DINOv3-paper {dataset} mask propagation")
    parser.add_argument("--video-split-manifest", type=Path, default=None)
    args = prepare_paths(parser.parse_args(), f"{dataset}_vos")
    if args.video_split_manifest is not None:
        args.video_split_manifests[dataset] = str(args.video_split_manifest.resolve())
    images, masks, names, split = preflight_masks(args.datasets_root, dataset, args.video_split_manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("Video evaluation requires an NVIDIA GPU")
    started, start_time = utc_now(), time.monotonic()
    identity = {**evaluation_identity(args), "checkpoint_fingerprint": checkpoint_fingerprint(args.checkpoint),
                "checkpoint_key": args.checkpoint_key, "video_resolution": args.video_resolution,
                "video_feature_blocks": args.video_feature_blocks,
                "video_split": split}
    progress_path = args.output_dir / "video_progress.json"
    videos = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("identity") == identity:
            videos = progress["videos"]
    backbone, metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    patch_size = metadata["patch_size"]
    short_side = RESOLUTIONS[patch_size][args.video_resolution]
    device = torch.device("cuda:0")
    backbone.to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    with torch.inference_mode():
        for index, name in enumerate(names, start=1):
            if name not in videos:
                frames, initialization = evaluation_frames(images / name, masks / name, dataset)
                objects, details = score_video(backbone, patch_size, frames,
                                               masks / name, short_side, device, dataset, args.video_feature_blocks)
                videos[name] = {**details, **initialization, "objects": objects}
                write_json(progress_path, {"identity": identity, "videos": videos})
            print_progress(f"{dataset} {args.video_resolution}", index, len(names))
    objects = [scores for video in videos.values() for scores in video["objects"].values()]
    if not objects:
        raise ValueError("No first-frame objects available for evaluation")
    j, f = (100 * float(np.mean([value[key] for value in objects])) for key in ("j", "f"))
    result = {
        "evaluation": f"{dataset}_vos", "task": "video_object_segmentation", "dataset": split["dataset"],
        "status": "completed", "metrics_status": "computed", "model": metadata,
        "evaluation_identity": evaluation_identity(args), "started_at": started, "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start_time,
        "metrics": {"j_and_f": (j + f) / 2, "j_mean": j, "f_mean": f},
        "protocol": {"source": PAPER, "sections": ["6.1", "6.1.5", "D.5", "Table 27"],
                     "implementation_details": NOTEBOOK, "resolution": args.video_resolution,
                     "short_side": short_side, "resize": "scale both native axes, independently round to nearest patch multiple",
                     "rgb_interpolation": "PIL bicubic", "feature_blocks": args.video_feature_blocks,
                     "feature": ("mean of last four LayerNorm-normalized patch-token blocks, then L2 normalized"
                                 if args.video_feature_blocks == 4 else "last normalized patch-token block, L2 normalized"),
                     "paper_feature_modification": args.video_feature_blocks != 1,
                     "past_frames": 7, "first_frame_always_present": True, "topk": 5, "temperature": .2,
                     "spatial_neighborhood": "unrestricted", "topk_ties": "include kth-rank ties",
                     "initial_mask_interpolation": "nearest-exact", "prediction_interpolation": "bilinear directly to native size",
                     "probability_postprocess": "released notebook per-channel min/max; NaN to zero",
                     "normalization_std": [.229, .224, .225], "first_frame_objects_only": True,
                     "later_gt_reseeding": False, "first_frame_scored": False,
                     "youtube_clip_initialization": "first provided annotated frame; leading unannotated RGB frames skipped",
                     "davis_last_frame_scored": False, "metric_resolution": "native",
                     "object_average": "mean over scored frames per object, then all objects",
                     "hyperparameters": "published DAVIS-train-selected Table 27 setting; no validation tuning",
                     "paper_split": split, "split_reproduction_verified": split["author_split_verified"],
                     "unreleased_details": ["custom split lists/seeds", "full benchmark scoring harness"],
                     "gpu": torch.cuda.get_device_name()},
        "videos": videos,
    }
    write_json(args.result_json, result)
    print(f"Completed {dataset}/{args.video_resolution}: {result['metrics']}", flush=True)


if __name__ == "__main__":
    raise SystemExit("Use evaluation.utils.davis_vos, youtube_vos_vos or mose_vos")
