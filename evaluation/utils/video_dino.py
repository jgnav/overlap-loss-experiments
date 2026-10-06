"""Original DINO DAVIS propagation at 480p, averaging the last four blocks.

The pinned upstream evaluator is kept in evaluation/vendor/dino. Only image
block aggregation changes its default numerical protocol. Chunking avoids
materializing the full context affinity matrix without changing its equations.
"""

from collections import deque
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from evaluation.utils.common import (
    base_parser, checkpoint_fingerprint, evaluation_identity, load_backbone,
    prepare_paths, print_progress, utc_now, write_json,
)
from evaluation.utils.video_dinov3 import dataset_layout, _frames
from evaluation.vendor.davis.metrics import db_eval_boundary, db_eval_iou


SOURCE = ("https://github.com/facebookresearch/dino/blob/"
          "4b96393c4c877d127cff9f077468e4a1cc2b5e2d/eval_video_segmentation.py")
INPUT_SIZE = 480
FEATURE_BLOCKS = 4
N_LAST_FRAMES = 7
NEIGHBORHOOD = 12
TOP_K = 5
TEMPERATURE = .1


def resized_size(native_wh, preserve_aspect=True):
    """DINO short-side 480, long side floored to a multiple of 64."""
    width, height = native_wh
    if not preserve_aspect:
        return INPUT_SIZE, INPUT_SIZE
    if height > width:
        return INPUT_SIZE, int((INPUT_SIZE * height / width // 64) * 64)
    return int((INPUT_SIZE * width / height // 64) * 64), INPUT_SIZE


def read_image(path, preserve_aspect=True):
    """Original DINO read_frame, including its std of 0.228."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read video RGB frame: {path}")
    native_wh = (image.shape[1], image.shape[0])
    image = cv2.resize(image, resized_size(native_wh, preserve_aspect)).astype(np.float32) / 255.
    image = torch.from_numpy(image[:, :, ::-1].transpose(2, 0, 1).copy()).float()
    for channel, mean, std in zip(image, (.485, .456, .406), (.228, .224, .225)):
        channel.sub_(mean).div_(std)
    return image, native_wh


@torch.no_grad()
def patch_features(backbone, image, patch_size):
    if any(side % patch_size for side in image.shape[-2:]):
        raise ValueError("DINO frame dimensions must be divisible by the patch size")
    layers = backbone.get_intermediate_layers(image[None], n=FEATURE_BLOCKS)
    if len(layers) != FEATURE_BLOCKS:
        raise ValueError("Backbone must return its last four normalized blocks")
    # The repository removes registers inside get_intermediate_layers.
    features = torch.stack([layer[0, 1:].float() for layer in layers]).mean(0)
    grid = tuple(side // patch_size for side in image.shape[-2:])
    if features.shape[0] != grid[0] * grid[1]:
        raise ValueError("Patch tokens do not match the image grid")
    return features, grid


def initial_probabilities(mask, grid, device):
    # Upstream sizes channels from the downsampled indexed mask, preserving
    # even empty intermediate IDs; never replace this with compact IDs.
    small = np.asarray(Image.fromarray(mask).resize((grid[1], grid[0]), Image.Resampling.NEAREST)).copy()
    labels = torch.from_numpy(small).to(device=device, dtype=torch.long)
    return F.one_hot(labels, num_classes=int(labels.max()) + 1).float().permute(2, 0, 1).flatten(1)


def propagate(target, features, probabilities, grid, radius=NEIGHBORHOOD,
              topk=TOP_K, temperature=TEMPERATURE, chunk_size=128):
    """Exact exp/mask/top-k/sum equations of DINO label_propagation."""
    if not features or len(features) != len(probabilities) or topk < 1 or temperature <= 0:
        raise ValueError("Invalid propagation context or hyperparameters")
    height, width = grid
    if len(target) != height * width or any(len(value) != len(target) for value in features):
        raise ValueError("Context and target grids must match")
    sources = F.normalize(torch.cat(features), dim=1)
    target = F.normalize(target, dim=1)
    masks = torch.cat(probabilities, dim=1)
    if masks.shape[1] != len(sources):
        raise ValueError("Context labels and features do not align")
    ys, xs = torch.meshgrid(torch.arange(height, device=target.device),
                           torch.arange(width, device=target.device), indexing="ij")
    coords = torch.stack((ys.flatten(), xs.flatten()), dim=1)
    source_coords = coords.repeat(len(features), 1)
    result = torch.empty((masks.shape[0], len(target)), device=target.device, dtype=target.dtype)
    for begin in range(0, len(target), chunk_size):
        end = min(begin + chunk_size, len(target))
        affinity = torch.exp((target[begin:end] @ sources.T) / temperature)
        if radius > 0:
            nearby = (coords[begin:end, None] - source_coords[None]).abs().amax(2) <= radius
            affinity *= nearby
        cutoff = affinity.topk(min(topk, len(sources)), dim=1).values[:, -1:]
        affinity[affinity < cutoff] = 0  # Includes ties, matching upstream.
        affinity /= affinity.sum(1, keepdim=True)
        result[:, begin:end] = masks @ affinity.T
    return result


def prediction(probabilities, grid, native_wh, patch_size):
    values = F.interpolate(probabilities.reshape(1, -1, *grid), scale_factor=patch_size,
                           mode="bilinear", align_corners=False, recompute_scale_factor=False)[0]
    # Preserve upstream norm_mask behavior, including constant positive channels.
    for channel in values:
        if channel.max() > 0:
            channel.sub_(channel.min())
            channel.div_(channel.max())
    labels = values.argmax(0).byte().cpu().numpy()
    return np.asarray(Image.fromarray(labels).resize(native_wh, Image.Resampling.NEAREST))


def preflight_masks(root, dataset, manifest_path=None):
    if dataset != "davis":
        raise ValueError("Original DINO releases a DAVIS protocol only; YouTube-VOS/MOSE require an explicit extension")
    if manifest_path is not None:
        raise ValueError("Original DINO uses the public DAVIS 2017 validation list")
    images, masks, names, details = dataset_layout(root, dataset)
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Missing DAVIS RGB/annotation roots: {images}, {masks}")
    for name in names:
        for frame in _frames(images / name):
            path = masks / name / f"{frame.stem}.png"
            if not path.is_file():
                raise FileNotFoundError(f"DAVIS scoring annotation missing: {path}")
    return images, masks, names, details


@torch.no_grad()
def score_video(backbone, patch_size, frames, mask_folder, device, prediction_folder, preserve_aspect=True):
    first_path = mask_folder / f"{frames[0].stem}.png"
    with Image.open(first_path) as source:
        first_mask = np.asarray(source).copy()
        palette = source.getpalette()
    object_ids = tuple(int(value) for value in np.unique(first_mask) if value not in (0, 255))
    if not object_ids:
        raise ValueError(f"First DAVIS annotation contains no objects: {first_path}")
    prediction_folder.mkdir(parents=True, exist_ok=True)
    def save_mask(name, mask):
        image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
        if palette is not None:
            image.putpalette(palette)
        image.save(prediction_folder / name)
    save_mask(first_path.name, first_mask)
    image, native_wh = read_image(frames[0], preserve_aspect)
    if first_mask.shape != (native_wh[1], native_wh[0]):
        raise ValueError("First RGB/annotation dimensions differ")
    first_features, grid = patch_features(backbone, image.to(device), patch_size)
    first_probs = initial_probabilities(first_mask, grid, device)
    history = deque(maxlen=N_LAST_FRAMES)
    scores = {str(value): [] for value in object_ids}
    for index, frame in enumerate(frames[1:], start=1):
        image, current_wh = read_image(frame, preserve_aspect)
        features, current_grid = patch_features(backbone, image.to(device), patch_size)
        if current_wh != native_wh or current_grid != grid:
            raise ValueError("Video frame dimensions changed")
        references = [(first_features, first_probs), *history]
        probs = propagate(features, [pair[0] for pair in references], [pair[1] for pair in references], grid)
        history.append((features, probs))  # Store soft patch labels before postprocess.
        predicted = prediction(probs, grid, native_wh, patch_size)
        save_mask(f"{frame.stem}.png", predicted)
        if index == len(frames) - 1:
            continue  # Official DAVIS scorer excludes first/last frames.
        with Image.open(mask_folder / f"{frame.stem}.png") as source:
            truth = np.asarray(source).copy()
        if truth.shape != first_mask.shape:
            raise ValueError(f"Scoring mask dimensions differ: {frame}")
        void = truth == 255
        for value in object_ids:
            gt, estimate = truth == value, predicted == value
            scores[str(value)].append((float(db_eval_iou(gt, estimate, void)),
                                      float(db_eval_boundary(gt, estimate, void))))
    if any(not values for values in scores.values()):
        raise ValueError("No scored DAVIS frames")
    return {key: {"j": float(np.mean([v[0] for v in values])),
                  "f": float(np.mean([v[1] for v in values]))} for key, values in scores.items()}, {
        "frames": len(frames), "scored_frames": len(frames) - 2, "object_ids": list(object_ids),
        "native_size_wh": list(native_wh), "input_size_wh": list(resized_size(native_wh, preserve_aspect)),
        "feature_grid_hw": list(grid), "prediction_folder": str(prediction_folder),
    }


def main(dataset):
    parser = base_parser("Original DINO DAVIS, 480p and last four blocks")
    parser.add_argument("--video-split-manifest", type=Path, default=None)
    args = prepare_paths(parser.parse_args(), f"{dataset}_vos")
    if args.video_resolution != "small" or args.video_feature_blocks != FEATURE_BLOCKS:
        raise ValueError("Original DINO protocols require video_resolution: small and video_feature_blocks: 4")
    preserve_aspect = args.video_protocol == "dino_480p_last4"
    images, masks, names, split = preflight_masks(args.datasets_root, dataset, args.video_split_manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("Video evaluation requires an NVIDIA GPU")
    started, start_time = utc_now(), time.monotonic()
    identity = {**evaluation_identity(args), "checkpoint_fingerprint": checkpoint_fingerprint(args.checkpoint),
                "checkpoint_key": args.checkpoint_key, "video_split": split}
    progress_path = args.output_dir / "video_progress.json"
    videos = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("identity") == identity:
            videos = progress["videos"]
    backbone, metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    device = torch.device("cuda:0")
    backbone.to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    with torch.inference_mode():
        for index, name in enumerate(names, start=1):
            if name not in videos:
                objects, details = score_video(backbone, metadata["patch_size"], _frames(images / name),
                                               masks / name, device, args.output_dir / "Annotations" / name, preserve_aspect)
                videos[name] = {**details, "objects": objects}
                write_json(progress_path, {"identity": identity, "videos": videos})
            print_progress(f"DAVIS {args.video_protocol}", index, len(names))
    objects = [scores for video in videos.values() for scores in video["objects"].values()]
    j, f = (100 * float(np.mean([value[key] for value in objects])) for key in ("j", "f"))
    vendor = Path(__file__).resolve().parents[1] / "vendor/dino/eval_video_segmentation.py"
    result = {
        "evaluation": "davis_vos", "task": "video_object_segmentation", "dataset": split["dataset"],
        "status": "completed", "metrics_status": "computed", "model": metadata,
        "evaluation_identity": evaluation_identity(args), "started_at": started, "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start_time,
        "metrics": {"j_and_f": (j + f) / 2, "j_mean": j, "f_mean": f},
        "protocol": {"source": SOURCE, "source_sha256": hashlib.sha256(vendor.read_bytes()).hexdigest(),
                     "name": args.video_protocol,
                     "modifications": (["mean of last four normalized blocks"] if preserve_aspect else
                                       ["480x480 RGB and annotation geometry", "mean of last four normalized blocks"]),
                     "short_side": INPUT_SIZE, "preserve_aspect": preserve_aspect,
                     "resize": ("DINO short side 480, long side rounded down to multiple of 64" if preserve_aspect else "480x480"),
                     "input_size": None if preserve_aspect else [INPUT_SIZE, INPUT_SIZE], "rgb_interpolation": "OpenCV INTER_LINEAR",
                     "normalization_mean": [.485, .456, .406], "normalization_std": [.228, .224, .225],
                     "feature_blocks": FEATURE_BLOCKS, "feature": "mean normalized blocks, then cosine normalization",
                     "past_frames": N_LAST_FRAMES, "first_frame_always_present": True,
                     "topk": TOP_K, "neighborhood_radius_patches": NEIGHBORHOOD, "temperature": TEMPERATURE,
                     "topk_ties": "include kth-rank ties", "initial_mask_interpolation": "PIL nearest to patch grid",
                     "prediction_interpolation": "bilinear by patch size, channel norm_mask, argmax, PIL nearest to native",
                     "soft_mask_history": True, "first_frame_scored": False, "davis_last_frame_scored": False,
                     "metric_resolution": "native", "object_average": "frames per object, then all objects",
                     "split": split, "gpu": torch.cuda.get_device_name()},
        "prediction_export": {"directory": str(args.output_dir / "Annotations"), "initialization_frame_included": True},
        "videos": videos,
    }
    write_json(args.result_json, result)
    print(f"Completed DAVIS {args.video_protocol}: {result['metrics']}", flush=True)
