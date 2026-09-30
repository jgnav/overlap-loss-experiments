"""DINO-style, training-free video mask propagation for CRISP's VOS table."""

import json
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms as T

from evaluation.utils.common import (
    base_parser, evaluation_identity, load_backbone, prepare_paths, print_progress,
    utc_now, write_json,
)
from evaluation.vendor.davis.metrics import db_eval_boundary, db_eval_iou


INPUT_SIZE = 480
N_LAST_FRAMES = 7
NEIGHBORHOOD = 12
TOP_K = 5
TEMPERATURE = 0.1
NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))
# Official YouTube-VOS 2019 scoring_program_release.zip/categories_list_seen.txt.
YOUTUBE_SEEN_CATEGORIES = frozenset("""airplane ape bear bike bird boat bucket bus camel cat cow crocodile
deer dog dolphin duck eagle earless_seal elephant fish fox frisbee frog giant_panda
giraffe hand hat hedgehog horse knife leopard lion lizard monkey motorbike mouse
owl paddle parachute parrot penguin person plant rabbit raccoon sedan shark sheep
sign skateboard snail snake snowboard squirrel surfboard tennis_racket tiger toilet
train truck turtle umbrella whale zebra""".split())


def _paths(root, dataset_name):
    choices = {
        "davis": ("davis2017", "DAVIS2017", "DAVIS"),
        "youtube_vos": ("youtube_vos_2019", "YouTubeVOS2019", "YouTube-VOS"),
        "mose": ("MOSEv2",),
    }[dataset_name]
    for name in choices:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"{dataset_name} directory missing under {root}; tried {choices}")


def _layout(root, dataset_name):
    if dataset_name == "davis":
        image_root, mask_root = root / "JPEGImages/480p", root / "Annotations/480p"
        split_file = root / "ImageSets/2017/val.txt"
    else:
        bases = (root / "valid", root / "val", root)
        candidate = next((base for base in bases if (base / "JPEGImages").is_dir()), None)
        if candidate is None:
            raise FileNotFoundError(f"Missing JPEGImages directory under {root}")
        image_root, mask_root = candidate / "JPEGImages", candidate / "Annotations"
        split_file = next((path for path in (
            candidate / "ImageSets/val.txt", candidate / "ImageSets/valid.txt",
            root / "ImageSets/val.txt", root / "ImageSets/valid.txt",
        ) if path.is_file()), None)
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Missing video RGB/annotation directories: {image_root}, {mask_root}")
    if split_file is not None and split_file.is_file():
        names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    elif dataset_name == "davis":
        raise FileNotFoundError(f"Missing official DAVIS 2017 validation list: {split_file}")
    else:
        names = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Empty or duplicate video split: {root}")
    return image_root, mask_root, names, split_file


def _frames(folder):
    files = sorted(path for path in folder.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(files) < 2:
        raise ValueError(f"Video needs at least two frames: {folder}")
    return files


def _youtube_metadata(image_root, names):
    path = image_root.parent / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Official YouTube-VOS validation metadata missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    videos = document.get("videos") if isinstance(document, dict) else None
    if not isinstance(videos, dict):
        raise ValueError(f"Invalid YouTube-VOS validation metadata: {path}")
    for name in names:
        video = videos.get(name)
        if not isinstance(video, dict) or not isinstance(video.get("objects"), dict) or not video["objects"]:
            raise ValueError(f"Missing YouTube-VOS object metadata for {name}")
        for object_id, info in video["objects"].items():
            frames = info.get("frames") if isinstance(info, dict) else None
            if (not object_id.isdecimal() or not info.get("category")
                    or not isinstance(frames, list) or len(frames) < 2
                    or any(not isinstance(frame, str) for frame in frames)
                    or len(frames) != len(set(frames))):
                raise ValueError(f"Invalid YouTube-VOS object metadata for {name}/{object_id}")
    return videos


def preflight_masks(root, dataset_name):
    """Check scoring masks, or initialization masks for MOSEv2 export."""
    dataset_root = _paths(root, dataset_name)
    image_root, mask_root, names, _ = _layout(dataset_root, dataset_name)
    youtube_videos = _youtube_metadata(image_root, names) if dataset_name == "youtube_vos" else None
    mose_videos = None
    if dataset_name == "mose":
        metadata_path = dataset_root / "meta_valid.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Official MOSEv2 validation metadata missing: {metadata_path}")
        mose_videos = json.loads(metadata_path.read_text(encoding="utf-8")).get("videos")
        if not isinstance(mose_videos, dict) or set(mose_videos) != set(names):
            raise ValueError("MOSEv2 RGB videos must match the official validation metadata")
    for name in names:
        frames = _frames(image_root / name)
        masks = mask_root / name
        frame_stems = {frame.stem for frame in frames}
        first = masks / f"{frames[0].stem}.png"
        if not first.is_file():
            raise FileNotFoundError(f"Initial validation mask missing: {first}")
        if dataset_name == "mose":
            info = mose_videos[name]
            if ([frame.name for frame in frames] != info.get("frames")
                    or len(frames) != info.get("length")):
                raise ValueError(f"MOSEv2 RGB frames do not match official metadata: {name}")
            initial = _annotation(first)
            if initial.shape != (info.get("height"), info.get("width")):
                raise ValueError(f"MOSEv2 initialization mask size differs from metadata: {first}")
            labels = set(np.unique(initial).tolist()) - {0, 255}
            if not labels or labels != set(info.get("objects", [])):
                raise ValueError(f"MOSEv2 initialization object IDs differ from metadata: {first}")
            continue
        if youtube_videos is not None:
            for object_id, info in youtube_videos[name]["objects"].items():
                for frame_id in info["frames"]:
                    if frame_id not in frame_stems:
                        raise FileNotFoundError(f"YouTube-VOS frame {frame_id} missing for {name}/{object_id}")
                    if not (masks / f"{frame_id}.png").is_file():
                        raise FileNotFoundError(f"YouTube-VOS scoring mask missing: {masks / f'{frame_id}.png'}")
        scored = frames[1:-1] if dataset_name == "davis" else frames[1:]
        available = sum((masks / f"{frame.stem}.png").is_file() for frame in scored)
        missing = len(scored) - available
        if available == 0:
            raise FileNotFoundError(f"No scoring masks in {masks}; first-frame masks alone cannot be scored offline")
        if dataset_name in ("davis", "mose") and missing:
            raise FileNotFoundError(f"{missing} validation scoring masks missing in {masks}")
        if dataset_name == "youtube_vos" and missing and available < 2:
            raise FileNotFoundError(f"Too few labeled YouTube-VOS scoring frames in {masks}")


def _annotation(path):
    with Image.open(path) as source:
        return np.asarray(source).copy()


def _save_mask(path, mask, palette):
    """Save indexed labels without converting object IDs into RGB colors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    image.putpalette(palette if palette is not None else [value for value in range(256) for _ in range(3)])
    image.save(path)


def _submission_archive(prediction_root, image_root, names, archive_path):
    """Package only this split's expected masks, including initialization frames."""
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            for frame in _frames(image_root / name):
                mask = prediction_root / name / f"{frame.stem}.png"
                if not mask.is_file():
                    raise FileNotFoundError(f"Prediction missing from submission: {mask}")
                archive.write(mask, arcname=mask.relative_to(prediction_root).as_posix())
    temporary.replace(archive_path)


def _read_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read video frame: {path}")
    original_size = (bgr.shape[1], bgr.shape[0])
    rgb = cv2.cvtColor(cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE)), cv2.COLOR_BGR2RGB)
    return NORMALIZE(T.ToTensor()(rgb)), original_size


@torch.no_grad()
def _features(backbone, image, patch_size, registers):
    layers = backbone.get_intermediate_layers(image[None], n=4)
    height, width = image.shape[-2] // patch_size, image.shape[-1] // patch_size
    representations = []
    for layer in layers:
        tokens = layer[:, 1:].float()
        if tokens.shape[1] != height * width:
            raise ValueError("Video feature grid does not match input/patch size")
        representations.append(tokens[0].T.reshape(-1, height, width))
    return F.normalize(torch.stack(representations).mean(0).flatten(1).T, dim=1), (height, width)


def _one_hot_mask(mask, object_ids, grid, device):
    labels = torch.as_tensor(mask.astype(np.int64), device=device)[None, None].float()
    # Nearest labels prevent fractional IDs; classes follow the initial mask.
    labels = F.interpolate(labels, size=grid, mode="nearest")[0, 0].long()
    return torch.stack([(labels == label).float() for label in (0, *object_ids)], dim=0).flatten(1)


def propagate_labels(target, sources, source_masks, grid,
                     radius=NEIGHBORHOOD, topk=TOP_K, temperature=TEMPERATURE):
    """DINO local top-k affinity over first and preceding frames."""
    height, width = grid
    count = height * width
    if len(sources) != len(source_masks) or not sources:
        raise ValueError("Source features/masks must be nonempty and aligned")
    source = torch.cat(sources, dim=0)
    masks = torch.cat(source_masks, dim=1)
    if source.shape[0] != masks.shape[1] or target.shape[0] != count:
        raise ValueError("Video source/target patch dimensions differ")
    # Compute per-target affinities without materializing a 4-D mask.
    y, x = torch.meshgrid(torch.arange(height, device=target.device),
                          torch.arange(width, device=target.device), indexing="ij")
    coordinates = torch.stack((y.flatten(), x.flatten()), dim=1)
    result = torch.empty((masks.shape[0], count), device=target.device)
    for start in range(0, count, 128):
        end = min(count, start + 128)
        similarity = target[start:end] @ source.T / temperature
        target_points = coordinates[start:end]
        source_points = coordinates.repeat(len(sources), 1)
        nearby = (target_points[:, None, :] - source_points[None]).abs().amax(2) <= radius
        similarity.masked_fill_(~nearby, float("-inf"))
        values, indices = similarity.topk(min(topk, nearby.shape[1]), dim=1)
        weights = values.softmax(1)
        result[:, start:end] = (masks[:, indices] * weights[None]).sum(-1)
    return result.reshape(1, masks.shape[0], height, width)


def _prediction(probabilities, original_size, object_ids):
    upsampled = F.interpolate(probabilities, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)[0]
    # Match DINO's per-channel min/max normalization before the argmax.
    flat = upsampled.flatten(1)
    minimum, maximum = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
    normalized = torch.where(maximum > minimum, (upsampled - minimum) / (maximum - minimum).clamp_min(1e-9), upsampled)
    labels = normalized.argmax(0).byte().cpu().numpy()
    mapping = np.asarray((0, *object_ids), dtype=np.uint8)
    native = Image.fromarray(mapping[labels]).resize(original_size, Image.Resampling.NEAREST)
    return np.asarray(native)


def _score_frame(predicted, truth, object_ids, youtube_official=False):
    if predicted.shape != truth.shape:
        raise ValueError("Predicted mask and ground truth have different sizes")
    valid = truth != 255
    scores = []
    for object_id in object_ids:
        gt, estimate = truth == object_id, predicted == object_id
        if youtube_official:
            # Official scorer computes J at native size and F after resizing
            # the short side to 360 pixels, with no DAVIS void-pixel mask.
            height, width = truth.shape
            scale = 360 / min(height, width)
            size = (int(width * scale), int(height * scale))
            gt_boundary = np.asarray(Image.fromarray(gt).resize(size, Image.Resampling.NEAREST))
            estimate_boundary = np.asarray(Image.fromarray(estimate).resize(size, Image.Resampling.NEAREST))
            scores.append((float(db_eval_iou(gt, estimate)),
                           float(db_eval_boundary(gt_boundary, estimate_boundary))))
        else:
            scores.append((float(db_eval_iou(gt, estimate, ~valid)),
                           float(db_eval_boundary(gt, estimate, ~valid))))
    return scores


def _score_video(backbone, metadata, frames, mask_folder, device, dataset_name, video_metadata=None,
                 prediction_folder=None):
    export_only = dataset_name == "mose"
    if export_only and prediction_folder is None:
        raise ValueError("MOSEv2 requires a prediction output directory")
    first_path = mask_folder / f"{frames[0].stem}.png"
    if not first_path.is_file():
        raise FileNotFoundError(f"Initial object mask missing: {first_path}")
    first_mask = _annotation(first_path)
    object_ids = tuple(int(value) for value in np.unique(first_mask) if value not in (0, 255))
    if not object_ids:
        raise ValueError(f"Initial mask has no foreground objects: {first_path}")
    frame, original_size = _read_image(frames[0])
    if first_mask.shape != (original_size[1], original_size[0]):
        raise ValueError(f"Initial mask and RGB frame sizes differ: {first_path}")
    palette = None
    if prediction_folder is not None:
        with Image.open(first_path) as source:
            palette = source.getpalette()
        _save_mask(prediction_folder / first_path.name, first_mask, palette)
    first_features, grid = _features(backbone, frame.to(device), metadata["patch_size"], metadata["num_register_tokens"])
    first_probabilities = _one_hot_mask(first_mask, object_ids, grid, device)
    history = deque(maxlen=N_LAST_FRAMES)
    scores = {object_id: [] for object_id in object_ids}
    scored_frames, missing_annotations = 0, []
    for index, frame_path in enumerate(frames[1:], start=1):
        frame, original_size = _read_image(frame_path)
        target_features, target_grid = _features(backbone, frame.to(device), metadata["patch_size"], metadata["num_register_tokens"])
        if target_grid != grid:
            raise ValueError("Video grid changed between frames")
        references = [(first_features, first_probabilities), *history]
        propagated = propagate_labels(target_features, [item[0] for item in references],
                                      [item[1] for item in references], grid)
        # YouTube-VOS may introduce new objects after frame zero. Their first
        # provided mask is an allowed reference, not a scored prediction.
        annotation_path = mask_folder / f"{frame_path.stem}.png"
        truth = _annotation(annotation_path) if not export_only and annotation_path.is_file() else None
        if truth is not None and dataset_name != "youtube_vos":
            unexpected = set(np.unique(truth).tolist()) - {0, 255, *object_ids}
            if unexpected:
                raise ValueError(f"Unexpected object IDs {sorted(unexpected)} in {annotation_path}")
        introduced = ()
        if dataset_name == "youtube_vos" and truth is not None:
            introduced = tuple(int(value) for value in np.unique(truth)
                               if value not in (0, 255, *object_ids))
            if introduced:
                pad = (0, 0, 0, len(introduced))
                first_probabilities = F.pad(first_probabilities, (0, 0, 0, len(introduced)))
                history = deque(((features, F.pad(probs, pad)) for features, probs in history),
                                maxlen=N_LAST_FRAMES)
                next_probabilities = F.pad(propagated[0].flatten(1), pad)
                truth_at_grid = torch.as_tensor(truth.astype(np.int64), device=device)[None, None].float()
                truth_at_grid = F.interpolate(truth_at_grid, size=grid, mode="nearest")[0, 0].long().flatten()
                for channel, object_id in enumerate(introduced, start=len(object_ids) + 1):
                    pixels = truth_at_grid == object_id
                    next_probabilities[:, pixels] = 0
                    next_probabilities[channel, pixels] = 1
                    scores[object_id] = []
                object_ids = (*object_ids, *introduced)
                propagated = next_probabilities.reshape(1, len(object_ids) + 1, *grid)
        predicted = _prediction(propagated, original_size, object_ids)
        if prediction_folder is not None:
            _save_mask(prediction_folder / f"{frame_path.stem}.png", predicted, palette)
        evaluate_frame = not export_only and not (dataset_name == "davis" and index == len(frames) - 1)
        if evaluate_frame:
            if truth is not None:
                scored_ids = object_ids
                if video_metadata is not None:
                    scored_ids = tuple(object_id for object_id in object_ids
                                       if frame_path.stem in video_metadata["objects"][str(object_id)]["frames"][1:])
                frame_scores = _score_frame(predicted, truth, scored_ids,
                                            youtube_official=video_metadata is not None)
                for object_id, value in zip(scored_ids, frame_scores):
                    if object_id not in introduced:
                        scores[object_id].append(value)
                scored_frames += 1
            else:
                missing_annotations.append(frame_path.name)
        history.append((target_features, propagated[0].flatten(1)))
    if not export_only and not scored_frames:
        raise ValueError(f"No scored mask frames in {mask_folder}")
    if not export_only and dataset_name in ("davis", "mose") and missing_annotations:
        raise FileNotFoundError(f"Missing validation masks in {mask_folder}: {missing_annotations[:5]}")
    if dataset_name == "youtube_vos" and missing_annotations and scored_frames < 2:
        raise ValueError(f"Too few labeled YouTube-VOS frames in {mask_folder}")
    object_scores = {str(object_id): {"j": float(np.mean([score[0] for score in values])),
                                      "f": float(np.mean([score[1] for score in values]))}
                     for object_id, values in scores.items() if values}
    return object_scores, {"frames": len(frames), "scored_frames": scored_frames,
                           "missing_annotations": missing_annotations, "objects": list(object_ids),
                           "exported_frames": len(frames) if prediction_folder is not None else 0}


def _youtube_metrics(per_video, videos):
    groups = {"seen": [], "unseen": []}
    for name, details in per_video.items():
        for object_id, scores in details["objects"].items():
            category = videos[name]["objects"][object_id]["category"]
            group = "seen" if category in YOUTUBE_SEEN_CATEGORIES else "unseen"
            groups[group].append(scores)
    if not groups["seen"] or not groups["unseen"]:
        raise ValueError("YouTube-VOS official scoring requires seen and unseen objects")
    metrics = {}
    for group, scores in groups.items():
        metrics[f"j_{group}"] = 100 * float(np.mean([score["j"] for score in scores]))
        metrics[f"f_{group}"] = 100 * float(np.mean([score["f"] for score in scores]))
    metrics["j_mean"] = (metrics["j_seen"] + metrics["j_unseen"]) / 2
    metrics["f_mean"] = (metrics["f_seen"] + metrics["f_unseen"]) / 2
    metrics["j_and_f"] = (metrics["j_mean"] + metrics["f_mean"]) / 2
    return metrics


def main(dataset_name):
    evaluation_name = f"{dataset_name}_vos"
    args = prepare_paths(base_parser(f"DINO-style {dataset_name} mask propagation").parse_args(), evaluation_name)
    if not torch.cuda.is_available():
        raise RuntimeError("Video mask propagation requires an NVIDIA GPU")
    started, start_time = utc_now(), time.monotonic()
    dataset_root = _paths(args.datasets_root, dataset_name)
    image_root, mask_root, names, split_file = _layout(dataset_root, dataset_name)
    preflight_masks(args.datasets_root, dataset_name)
    youtube_videos = _youtube_metadata(image_root, names) if dataset_name == "youtube_vos" else None
    export_only = dataset_name == "mose"
    prediction_root = args.output_dir / "Annotations" if export_only else None
    backbone, metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    device = torch.device("cuda:0")
    backbone.to(device).eval()
    all_objects, per_video = [], {}
    with torch.inference_mode():
        for index, name in enumerate(names, start=1):
            frames = _frames(image_root / name)
            object_scores, details = _score_video(backbone, metadata, frames, mask_root / name, device,
                                                  dataset_name, youtube_videos[name] if youtube_videos else None,
                                                  prediction_folder=prediction_root / name if prediction_root else None)
            all_objects.extend(object_scores.values())
            per_video[name] = {**details, "object_ids": details["objects"], "objects": object_scores}
            print_progress(f"{dataset_name} video", index, len(names))
    if not export_only and not all_objects:
        raise ValueError(f"No annotated objects in {dataset_name} evaluation")
    if export_only:
        archive_path = args.output_dir / "mosev2_valid_submission.zip"
        _submission_archive(prediction_root, image_root, names, archive_path)
        metrics = {"j_and_f": None, "j_mean": None, "f_mean": None}
    elif youtube_videos is not None:
        metrics = _youtube_metrics(per_video, youtube_videos)
    else:
        j = 100 * float(np.mean([score["j"] for score in all_objects]))
        f = 100 * float(np.mean([score["f"] for score in all_objects]))
        metrics = {"j_and_f": (j + f) / 2, "j_mean": j, "f_mean": f}
    write_json(args.result_json, {
        "evaluation": evaluation_name, "task": "video_object_segmentation",
        "dataset": {"davis": "DAVIS 2017 val", "youtube_vos": "YouTube-VOS 2019 val", "mose": "MOSEv2 val"}[dataset_name],
        "status": "completed", "started_at": started, "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start_time,
        "model": metadata, "evaluation_identity": evaluation_identity(args),
        "metrics_status": "pending_external_evaluation" if export_only else "computed",
        "prediction_export": {"directory": str(prediction_root), "archive": str(archive_path),
                              "frames": sum(item["exported_frames"] for item in per_video.values()),
                              "initialization_frame_included": True,
                              "server": "https://www.codabench.org/competitions/10062/"} if export_only else None,
        "protocol": {"source": "CRISP Appendix A.2 and DINO video mask propagation",
                     "input_size": [INPUT_SIZE, INPUT_SIZE], "feature": "mean of last four LayerNorm-normalized patch-token blocks",
                     "reference": "first annotated frame plus preceding seven propagated frames",
                     "topk": TOP_K, "neighborhood_radius_patches": NEIGHBORHOOD,
                     "temperature": TEMPERATURE, "split_list": str(split_file) if split_file else None,
                     "dataset_root": str(dataset_root), "object_average": "mean over frame scores per object, then objects",
                     "mode": "prediction_export" if export_only else "offline_scoring",
                     "first_frame_scored": False, "davis_last_frame_scored": False,
                     "youtube_new_objects": "first annotated appearance used as reference, excluded from its own score",
                     "youtube_scoring": "official meta.json object frames, 360-pixel boundary F, seen/unseen four-metric mean" if youtube_videos else None},
        "videos": per_video, "metrics": metrics,
    })
    print(f"Completed {evaluation_name}: {metrics}", flush=True)
