"""Fixed, lightweight representation probes for the training loop.

The online probes intentionally use one protocol (k=20, cosine similarity)
instead of the full evaluation sweeps.  They are also usable as a standalone
module by pointing them at an immutable training checkpoint::

    python -m evaluation.online_probes --checkpoint teacher_epoch0010.pth \
        --datasets-root dataset --output probes/epoch0010.json

Training launches this module asynchronously through ``OnlineProbeRunner``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms as T

from evaluation.utils.common import load_backbone, utc_now
from evaluation.utils.datasets import make_pascal_voc
from evaluation.utils.imagenet import _resolve_imagenet_root


ONLINE_K = 20
ONLINE_FREQUENCY = 10
IMAGENET_TRAIN_SIZE = 10_000
IMAGENET_VAL_SIZE = 5_000
VOC_TRAIN_SIZE = 400
VOC_VAL_SIZE = 200
VOC_RESOLUTION = 256
VOC_PATCH_GRID = 16
VOC_IGNORE_LABEL = 255

IMAGENET_NORMALIZE = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def probe_due(epoch, frequency=ONLINE_FREQUENCY):
    """Return whether one-based completed ``epoch`` should be evaluated."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if type(frequency) is not int or frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    return epoch % frequency == 0


def select_stratified_indices(targets, size, seed):
    """Select a fixed-size proportional class-stratified subset.

    Class quotas use largest remainders, then each class is sampled with a
    private deterministic generator.  The final order is shuffled with that
    same generator, so the result is independent of DataLoader workers.
    """
    targets = torch.as_tensor(targets, dtype=torch.long).flatten()
    if targets.numel() == 0:
        raise ValueError("Cannot select from an empty dataset")
    if type(size) is not int or not 1 <= size <= len(targets):
        raise ValueError(f"subset size must be an integer in [1, {len(targets)}]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    classes = torch.unique(targets, sorted=True)
    class_indices = [torch.where(targets == label)[0] for label in classes]
    quotas_float = [size * len(indices) / len(targets) for indices in class_indices]
    quotas = [math.floor(quota) for quota in quotas_float]
    remainder = size - sum(quotas)
    order = sorted(
        range(len(classes)),
        key=lambda i: (quotas_float[i] - quotas[i], -i),
        reverse=True,
    )
    for i in order[:remainder]:
        quotas[i] += 1
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for indices, quota in zip(class_indices, quotas, strict=True):
        permutation = torch.randperm(len(indices), generator=generator)[:quota]
        selected.extend(indices[permutation].tolist())
    permutation = torch.randperm(len(selected), generator=generator)
    return [selected[i] for i in permutation.tolist()]


def select_fixed_indices(length, size, seed):
    """Select a reproducible uniform subset when no image-level labels exist."""
    if type(length) is not int or length < 1:
        raise ValueError("dataset length must be a positive integer")
    if type(size) is not int or not 1 <= size <= length:
        raise ValueError(f"subset size must be an integer in [1, {length}]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:size].tolist()


def _indices_hash(indices):
    values = torch.as_tensor(indices, dtype=torch.int64).numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _validate_k(k):
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")


def _cosine_neighbor_rows(train_features, train_labels, query_features, k, chunk_size=256):
    """Return top-k labels/similarities without materializing all pairwise scores."""
    _validate_k(k)
    if train_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("Features must be two-dimensional")
    if train_features.shape[1] != query_features.shape[1]:
        raise ValueError("Train and query feature dimensions differ")
    if len(train_features) == 0 or len(query_features) == 0:
        raise ValueError("k-NN requires nonempty train and query features")
    if k > len(train_features):
        raise ValueError(f"k={k} exceeds the {len(train_features)} training features")
    train_labels = torch.as_tensor(train_labels, dtype=torch.long)
    if len(train_labels) != len(train_features):
        raise ValueError("Training labels and features have different lengths")
    train_features = F.normalize(train_features.float(), dim=1)
    query_features = F.normalize(query_features.float(), dim=1)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    values, neighbors = [], []
    for start in range(0, len(query_features), chunk_size):
        similarity = query_features[start:start + chunk_size] @ train_features.T
        chunk_values, chunk_neighbors = similarity.topk(k, dim=1, largest=True, sorted=True)
        values.append(chunk_values)
        neighbors.append(chunk_neighbors)
    values = torch.cat(values)
    labels = train_labels[torch.cat(neighbors)]
    return labels, values


def _cosine_neighbors(train_features, train_labels, query_features, k):
    """Predict labels by deterministic cosine k-NN majority vote."""
    labels, values = _cosine_neighbor_rows(train_features, train_labels, query_features, k)
    return _vote_predictions(labels, values, train_labels)


def _vote_predictions(labels, values, train_labels):
    classes = torch.unique(train_labels, sorted=True)
    # Vote count is primary; summed similarity and class index make ties stable.
    predictions = []
    for row_labels, row_values in zip(labels, values, strict=True):
        scores = []
        for label in classes:
            mask = row_labels == label
            scores.append((int(mask.sum()), float(row_values[mask].sum()), -int(label)))
        winner = max(zip(scores, classes.tolist(), strict=True))[1]
        predictions.append(winner)
    return torch.tensor(predictions, dtype=torch.long)


def knn_classification_metrics(
    train_features, train_labels, query_features, query_labels, k=ONLINE_K
):
    """Return top-1/top-5 percentages for a fixed cosine k-NN bank."""
    neighbor_labels, neighbor_similarities = _cosine_neighbor_rows(
        train_features, train_labels, query_features, k
    )
    predictions = _vote_predictions(neighbor_labels, neighbor_similarities, train_labels)
    query_labels = torch.as_tensor(query_labels, dtype=torch.long)
    if len(predictions) != len(query_labels):
        raise ValueError("Query labels and features have different lengths")
    classes = torch.unique(torch.as_tensor(train_labels, dtype=torch.long), sorted=True)
    # A k-NN vote produces one class.  Report top-5 from vote/similarity ranking
    # so top-5 remains meaningful even though top-1 uses the winning vote.
    rankings = []
    for row, row_sim in zip(neighbor_labels, neighbor_similarities, strict=True):
        ranking = sorted(
            classes.tolist(),
            key=lambda label: (
                int((row == label).sum()),
                float(row_sim[row == label].sum()),
                -int(label),
            ),
            reverse=True,
        )
        rankings.append(ranking[:5])
    rankings = torch.as_tensor(rankings, dtype=torch.long)
    return {
        "top1": 100.0 * (predictions == query_labels).float().mean().item(),
        "top5": 100.0 * rankings.eq(query_labels[:, None]).any(dim=1).float().mean().item(),
    }


def _center_crop_transform(size=224):
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        IMAGENET_NORMALIZE,
    ])


class _IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, label


@torch.inference_mode()
def _extract_image_features(model, dataset, batch_size, num_workers, dense=False, device=None):
    if len(dataset) == 0:
        raise ValueError("Probe dataset is empty")
    device = device or next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    features, labels = [], []
    for images, targets in loader:
        tokens = model.get_intermediate_layers(images.to(device, non_blocking=True), n=1)[0]
        tokens = tokens[:, 1:] if dense else tokens[:, 0]
        features.append(tokens.float().cpu())
        labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def _imagenet_datasets(datasets_root, train_size, val_size, seed):
    root = _resolve_imagenet_root(Path(datasets_root))
    transform = _center_crop_transform(224)
    full_train = datasets.ImageFolder(root / "train", transform=transform)
    full_val = datasets.ImageFolder(root / "val", transform=transform)
    if full_train.class_to_idx != full_val.class_to_idx or len(full_train.classes) != 1000:
        raise ValueError("ImageNet train/val must share exactly the same 1,000 classes")
    train_indices = select_stratified_indices(full_train.targets, train_size, seed)
    val_indices = select_stratified_indices(full_val.targets, val_size, seed + 1)
    return (
        _IndexedSubset(full_train, train_indices),
        _IndexedSubset(full_val, val_indices),
        {
            "train": len(train_indices),
            "validation": len(val_indices),
            "classes": 1000,
            "train_indices_sha256": _indices_hash(train_indices),
            "validation_indices_sha256": _indices_hash(val_indices),
        },
    )


def run_imagenet_probe(
    model,
    datasets_root,
    train_size=IMAGENET_TRAIN_SIZE,
    val_size=IMAGENET_VAL_SIZE,
    seed=0,
    k=ONLINE_K,
    batch_size=256,
    num_workers=0,
    device=None,
):
    """Run the fixed ImageNet CLS probe on an already loaded teacher model."""
    train, validation, metadata = _imagenet_datasets(
        datasets_root, train_size, val_size, seed
    )
    train_features, train_labels = _extract_image_features(
        model, train, batch_size, num_workers, device=device
    )
    val_features, val_labels = _extract_image_features(
        model, validation, batch_size, num_workers, device=device
    )
    return {
        **knn_classification_metrics(train_features, train_labels, val_features, val_labels, k),
        "dataset": "ImageNet-1K",
        "feature": "teacher final normalized CLS token",
        "input_resolution": 224,
        "k": k,
        "distance": "cosine",
        "subset": metadata,
    }


def _dense_transforms(resolution=VOC_RESOLUTION):
    image = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        IMAGENET_NORMALIZE,
    ])

    def target(mask):
        resized = T.functional.resize(
            mask.convert("P"), (resolution, resolution), interpolation=T.InterpolationMode.NEAREST
        )
        return torch.from_numpy(np.array(resized, copy=True)).long()

    return image, target


def _patch_labels(labels, grid=VOC_PATCH_GRID):
    if labels.ndim != 3 or labels.shape[1] % grid or labels.shape[2] % grid:
        raise ValueError("VOC masks must be divisible by the patch grid")
    height, width = labels.shape[1] // grid, labels.shape[2] // grid
    patches = labels.reshape(labels.shape[0], grid, height, grid, width).permute(0, 1, 3, 2, 4)
    patches = patches.reshape(-1, height * width)
    result = torch.full((len(patches),), VOC_IGNORE_LABEL, dtype=torch.long)
    for index, row in enumerate(patches):
        row = row[row != VOC_IGNORE_LABEL]
        if len(row):
            result[index] = torch.bincount(row, minlength=21).argmax()
    return result


@torch.inference_mode()
def _extract_dense_features(
    model, dataset, batch_size, num_workers, device=None, include_patch_labels=True
):
    if len(dataset) == 0:
        raise ValueError("VOC probe dataset is empty")
    device = device or next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
        persistent_workers=num_workers > 0,
    )
    features, labels = [], []
    for images, targets in loader:
        tokens = model.get_intermediate_layers(images.to(device, non_blocking=True), n=1)[0][:, 1:]
        grid = math.isqrt(tokens.shape[1])
        if grid != VOC_PATCH_GRID or grid * grid != tokens.shape[1]:
            raise ValueError(f"Expected 16 x 16 final patch tokens, got {tokens.shape[1]}")
        features.append(tokens.float().cpu().reshape(-1, tokens.shape[-1]))
        if include_patch_labels:
            labels.append(_patch_labels(targets))
    return torch.cat(features), torch.cat(labels) if labels else None


def _dense_pixel_metrics(predicted_patches, targets, resolution=VOC_RESOLUTION):
    grid = VOC_PATCH_GRID
    if type(resolution) is not int or resolution <= 0 or resolution % grid:
        raise ValueError("resolution must be a positive multiple of the 16 x 16 patch grid")
    targets = targets.long()
    if targets.ndim != 3 or targets.shape[1:] != (resolution, resolution):
        raise ValueError("VOC targets must have shape [images, resolution, resolution]")
    if len(predicted_patches) != len(targets) * grid * grid:
        raise ValueError("Predicted patch count does not match VOC target images")
    patch_size = resolution // grid
    prediction = predicted_patches.reshape(-1, grid, grid).repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
    target = targets
    valid = target != VOC_IGNORE_LABEL
    if not valid.any():
        raise ValueError("VOC validation subset has no valid pixels")
    correct = (prediction == target) & valid
    intersection, union = [], []
    for class_id in range(21):
        predicted = prediction == class_id
        actual = target == class_id
        intersection.append((predicted & actual & valid).sum())
        union.append((predicted | actual) & valid)
    union = torch.stack([item.sum() for item in union])
    intersection = torch.stack(intersection)
    present = union > 0
    miou = (intersection[present].float() / union[present].float()).mean().item()
    return {
        "miou": miou,
        "miou_percent": 100.0 * miou,
        "pixel_accuracy": correct.sum().float().div(valid.sum()).item(),
        "pixel_accuracy_percent": 100.0 * correct.sum().float().div(valid.sum()).item(),
    }


def _voc_datasets(datasets_root, train_size, val_size, seed):
    image_transform, target_transform = _dense_transforms()
    full_train = make_pascal_voc(
        Path(datasets_root), "train", image_transform, target_transform
    )
    full_val = make_pascal_voc(
        Path(datasets_root), "val", image_transform, target_transform
    )
    train_indices = select_fixed_indices(len(full_train), train_size, seed)
    val_indices = select_fixed_indices(len(full_val), val_size, seed + 1)
    return (
        Subset(full_train, train_indices),
        Subset(full_val, val_indices),
        {
            "train": len(train_indices),
            "validation": len(val_indices),
            "train_indices_sha256": _indices_hash(train_indices),
            "validation_indices_sha256": _indices_hash(val_indices),
            "patch_grid": "16x16",
            "classes": 21,
        },
    )


def run_voc_dense_probe(
    model,
    datasets_root,
    train_size=VOC_TRAIN_SIZE,
    val_size=VOC_VAL_SIZE,
    seed=0,
    k=ONLINE_K,
    batch_size=32,
    num_workers=0,
    device=None,
):
    """Run fixed VOC dense patch k-NN and score tiled patch predictions."""
    metadata = {"input_resolution": VOC_RESOLUTION, "feature": "teacher final normalized patch tokens"}
    train, validation, sizes = _voc_datasets(datasets_root, train_size, val_size, seed)
    train_features, train_labels = _extract_dense_features(
        model, train, batch_size, num_workers, device=device
    )
    valid_train = train_labels != VOC_IGNORE_LABEL
    if not valid_train.any():
        raise ValueError("VOC training subset has no valid patch labels")
    validation_features, _ = _extract_dense_features(
        model, validation, batch_size, num_workers, device=device, include_patch_labels=False
    )
    predicted = _cosine_neighbors(
        train_features[valid_train], train_labels[valid_train], validation_features, k
    )
    # Re-load validation masks only through the already transformed subset; the
    # feature extractor intentionally keeps only patch labels, so reconstruct
    # the pixel labels in one bounded pass for scoring.
    pixel_targets = []
    for _, targets in DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        pixel_targets.append(targets)
    pixel_targets = torch.cat(pixel_targets)
    return {
        **_dense_pixel_metrics(predicted, pixel_targets),
        "dataset": "PASCAL VOC 2012",
        "feature": "teacher final normalized patch tokens",
        "input_resolution": VOC_RESOLUTION,
        "k": k,
        "distance": "cosine",
        "subset": {**sizes, **metadata},
    }


def run_probe_checkpoint(
    checkpoint,
    datasets_root,
    output,
    arch="auto",
    seed=0,
    imagenet_train_size=IMAGENET_TRAIN_SIZE,
    imagenet_val_size=IMAGENET_VAL_SIZE,
    voc_train_size=VOC_TRAIN_SIZE,
    voc_val_size=VOC_VAL_SIZE,
    k=ONLINE_K,
    batch_size=256,
    num_workers=0,
    epoch=None,
    frequency=ONLINE_FREQUENCY,
):
    """Load one immutable teacher checkpoint and run both fixed probes."""
    _validate_k(k)
    if type(frequency) is not int or frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    started = utc_now()
    timer = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_metadata = load_backbone(checkpoint, "teacher", arch)
    model.to(device).eval()
    if model_metadata["patch_size"] != 16:
        raise ValueError(
            "The online VOC dense probe requires a ViT-S/16-compatible patch size of 16"
        )
    imagenet = run_imagenet_probe(
        model, datasets_root, imagenet_train_size, imagenet_val_size,
        seed, k, batch_size, num_workers, device,
    )
    voc = run_voc_dense_probe(
        model, datasets_root, voc_train_size, voc_val_size,
        seed, k, min(batch_size, 32), num_workers, device,
    )
    result = {
        "evaluation": "online_probes",
        "status": "completed",
        "epoch": epoch,
        "started_at": started,
        "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - timer,
        "checkpoint": str(Path(checkpoint).resolve()),
        "model": model_metadata,
        "protocol": {
            "teacher_checkpoint_key": "teacher",
            "seed": seed,
            "k": k,
            "distance": "cosine",
            "frequency": frequency,
            "imagenet": "fixed stratified train/validation subsets, center crop 224",
            "voc": "fixed train/validation subsets, resize 256, final 16x16 patch tokens",
        },
        "imagenet_cls_knn": imagenet,
        "voc_dense_knn": voc,
        "metrics": {
            "imagenet_cls_knn": imagenet,
            "voc_dense_knn": voc,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return result


def immutable_checkpoint_copy(source, destination):
    """Copy a completed checkpoint through a same-directory atomic rename."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


class OnlineProbeRunner:
    """Launch bounded asynchronous probes from immutable epoch snapshots."""

    def __init__(self, args, repository_root=None):
        self.args = args
        self.repository_root = Path(repository_root or Path(__file__).resolve().parents[1])
        self.processes = []
        self.submitted_results = []
        self.completed_results = set()

    def _reap(self):
        active = []
        for process, log in self.processes:
            if process.poll() is None:
                active.append((process, log))
            else:
                log.close()
        self.processes = active

    def submit(self, epoch, checkpoint):
        """Snapshot and launch a probe for completed one-based ``epoch``."""
        if not getattr(self.args, "online_probes_enabled", False):
            return None
        if not probe_due(epoch, self.args.online_probe_frequency):
            return None
        self._reap()
        maximum = self.args.online_probe_max_concurrent_jobs
        root = Path(self.args.output_dir) / "online_probes"
        # The copy is made even when all worker slots are occupied, so every
        # scheduled epoch remains an immutable, retryable teacher snapshot.
        snapshot = immutable_checkpoint_copy(
            checkpoint, root / "checkpoints" / f"teacher_epoch{epoch:04d}.pth"
        )
        if len(self.processes) >= maximum:
            print(
                f"Online probes: skipping epoch {epoch}; {maximum} job(s) still running "
                f"(snapshot retained at {snapshot})",
                flush=True,
            )
            return {"epoch": epoch, "checkpoint": snapshot, "status": "skipped"}
        result = root / f"epoch{epoch:04d}.json"
        log_path = root / "logs" / f"epoch{epoch:04d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "evaluation.online_probes",
            "--checkpoint", str(snapshot), "--datasets-root", str(self.args.online_probe_datasets_root),
            "--output", str(result), "--arch", self.args.arch, "--seed", str(self.args.seed),
            "--imagenet-train-size", str(self.args.online_probe_imagenet_train_size),
            "--imagenet-val-size", str(self.args.online_probe_imagenet_val_size),
            "--voc-train-size", str(self.args.online_probe_voc_train_size),
            "--voc-val-size", str(self.args.online_probe_voc_val_size),
            "--k", str(self.args.online_probe_k), "--batch-size", str(self.args.online_probe_batch_size),
            "--num-workers", str(self.args.online_probe_num_workers), "--epoch", str(epoch),
            "--frequency", str(self.args.online_probe_frequency),
        ]
        environment = os.environ.copy()
        gpu = getattr(self.args, "online_probe_gpu", None)
        if gpu not in (None, "", "none", "None"):
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=self.repository_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append((process, log))
        self.submitted_results.append((epoch, result))
        print(f"Online probes: launched epoch {epoch} from {snapshot}", flush=True)
        return {"epoch": epoch, "checkpoint": snapshot, "result": result, "pid": process.pid}

    def collect_completed(self):
        """Return metrics from newly finished jobs in training-stat form.

        A job that is still running is left for the next epoch.  Failed jobs
        remain visible in their per-epoch log and do not fabricate metrics.
        """
        self._reap()
        metrics = {}
        completed = getattr(self, "completed_results", set())
        for _, result_path in getattr(self, "submitted_results", []):
            if result_path in completed or not result_path.is_file():
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if result.get("status") != "completed":
                completed.add(result_path)
                continue
            image = result.get("imagenet_cls_knn", {})
            dense = result.get("voc_dense_knn", {})
            metrics.update({
                "online_imagenet_cls_knn_top1": float(image["top1"]),
                "online_imagenet_cls_knn_top5": float(image["top5"]),
                "online_voc_dense_knn_miou": float(dense["miou"]),
                "online_voc_dense_knn_miou_percent": float(dense["miou_percent"]),
                "online_voc_dense_knn_pixel_accuracy": float(dense["pixel_accuracy"]),
                "online_voc_dense_knn_pixel_accuracy_percent": float(
                    dense["pixel_accuracy_percent"]
                ),
                "online_probe_completed_epoch": int(result["epoch"]),
            })
            completed.add(result_path)
        self.completed_results = completed
        return metrics

    def close(self, wait=False):
        """Reap completed jobs; optionally wait for all outstanding probes."""
        if wait:
            for process, _ in self.processes:
                process.wait()
        for _, log in self.processes:
            log.close()
        self.processes = []


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="auto", choices=("auto", "vit_small", "vit_base", "vit_large"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imagenet-train-size", type=int, default=IMAGENET_TRAIN_SIZE)
    parser.add_argument("--imagenet-val-size", type=int, default=IMAGENET_VAL_SIZE)
    parser.add_argument("--voc-train-size", type=int, default=VOC_TRAIN_SIZE)
    parser.add_argument("--voc-val-size", type=int, default=VOC_VAL_SIZE)
    parser.add_argument("--k", type=int, default=ONLINE_K)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--frequency", type=int, default=ONLINE_FREQUENCY)
    return parser


def main():
    args = _parser().parse_args()
    try:
        run_probe_checkpoint(
            args.checkpoint, args.datasets_root, args.output, args.arch, args.seed,
            args.imagenet_train_size, args.imagenet_val_size, args.voc_train_size,
            args.voc_val_size, args.k, args.batch_size, args.num_workers, args.epoch,
            args.frequency,
        )
    except Exception as error:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "evaluation": "online_probes",
            "status": "failed",
            "epoch": args.epoch,
            "checkpoint": str(args.checkpoint),
            "error": f"{type(error).__name__}: {error}",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        raise


if __name__ == "__main__":
    main()
