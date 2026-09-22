import hashlib
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms as T

from evaluation.utils.common import (
    base_parser,
    cleanup_distributed,
    initialize_distributed,
    is_main_process,
    launch_distributed_if_needed,
    load_backbone,
    prepare_paths,
    print_progress,
    utc_now,
    write_json,
    evaluation_identity,
)


IMAGENET_NORMALIZE = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
BATCH_SIZE_PER_GPU = 256
IMAGENET_KNN_TRAINING_FRACTION = 0.10
IMAGENET_KNN_FRACTIONS = {
    "imagenet_knn_1pct": 0.01,
    "imagenet_knn": IMAGENET_KNN_TRAINING_FRACTION,
    "imagenet_knn_100pct": 1.0,
}
SIMCLRV2_SUBSET_URL = (
    "https://github.com/google-research/simclr/tree/master/imagenet_subsets"
)
SIMCLRV2_SUBSET_FILES = {
    0.01: "1percent.txt",
    0.10: "10percent.txt",
}
SIMCLRV2_SUBSET_SHA256 = {
    0.01: "71e82a48ba78252683ae334c4d019ebd5c6e855b9599c1389fbe83eb9549cf17",
    0.10: "6d09de11e7bdaf5b1f3b1f249b6183695f97310cdd20f0c03e7235b6b9392091",
}
SIMCLRV2_SUBSET_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "resources" / "simclrv2_imagenet_subsets"
)


def _resolve_imagenet_root(datasets_root):
    candidates = [
        datasets_root / "imagenet",
        datasets_root / "ImageNet",
        datasets_root / "imagenet-1k",
        datasets_root / "ILSVRC2012",
        datasets_root,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find ImageNet-1K train/ and val/ directories below "
        f"{datasets_root}"
    )


class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index


class IndexedSubset(torch.utils.data.Dataset):
    """Subset that exposes contiguous indices for distributed feature storage."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, label, index


def _simclrv2_subset_names(fraction):
    """Return the official fixed SimCLRv2 ImageNet image-name list."""
    try:
        filename = SIMCLRV2_SUBSET_FILES[fraction]
    except KeyError as error:
        raise ValueError(f"No SimCLRv2 subset exists for fraction {fraction}") from error
    path = SIMCLRV2_SUBSET_DIRECTORY / filename
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    expected_digest = SIMCLRV2_SUBSET_SHA256[fraction]
    if digest != expected_digest:
        raise RuntimeError(
            f"SimCLRv2 subset file checksum mismatch for {path}: "
            f"expected {expected_digest}, got {digest}"
        )
    names = tuple(line.strip() for line in contents.decode("utf-8").splitlines() if line.strip())
    if len(names) != len(set(names)):
        raise RuntimeError(f"SimCLRv2 subset file contains duplicate image names: {path}")
    return names


def _simclrv2_subset_indices(dataset, fraction):
    """Resolve a supplied SimCLRv2 image-name list against ImageFolder samples."""
    names = _simclrv2_subset_names(fraction)
    indices_by_name = {}
    for index, (sample, _) in enumerate(dataset.samples):
        name = Path(sample).name
        if name in indices_by_name:
            raise RuntimeError(
                "ImageNet training set has duplicate image basenames; cannot "
                f"unambiguously resolve the SimCLRv2 split ({name})"
            )
        indices_by_name[name] = index
    missing = [name for name in names if name not in indices_by_name]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"SimCLRv2 {round(fraction * 100)}% ImageNet split is missing "
            f"{len(missing)} images below {dataset.root}; first missing: {preview}"
        )
    return [indices_by_name[name] for name in names]


def _simclrv2_subset_protocol(fraction):
    filename = SIMCLRV2_SUBSET_FILES[fraction]
    return {
        "training_subset": f"official SimCLRv2 {round(fraction * 100)}% ImageNet split",
        "training_subset_file": f"evaluation/resources/simclrv2_imagenet_subsets/{filename}",
        "training_subset_file_sha256": SIMCLRV2_SUBSET_SHA256[fraction],
        "training_subset_source": SIMCLRV2_SUBSET_URL,
    }


def _indices_sha256(indices):
    values = torch.as_tensor(indices, dtype=torch.int64).numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _eval_transform():
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            IMAGENET_NORMALIZE,
        ]
    )


def _feature_vector(model, images):
    return model.get_intermediate_layers(images, n=1)[0][:, 0]


@torch.inference_mode()
def _extract_distributed_features(model, dataset, args, description):
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    features = None
    labels = None
    if is_main_process():
        print(f"{description}: starting {len(loader)} batches", flush=True)
    for batch_index, (images, batch_labels, indices) in enumerate(loader, start=1):
        images = images.cuda(non_blocking=True)
        batch_labels = batch_labels.cuda(non_blocking=True).contiguous()
        indices = indices.cuda(non_blocking=True).contiguous()
        batch_features = _feature_vector(model, images).float().contiguous()
        gathered_features = [
            torch.empty_like(batch_features) for _ in range(dist.get_world_size())
        ]
        gathered_labels = [
            torch.empty_like(batch_labels) for _ in range(dist.get_world_size())
        ]
        gathered_indices = [
            torch.empty_like(indices) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_features, batch_features)
        dist.all_gather(gathered_labels, batch_labels)
        dist.all_gather(gathered_indices, indices)
        all_features = torch.cat(gathered_features)
        all_labels = torch.cat(gathered_labels)
        all_indices = torch.cat(gathered_indices)
        if features is None:
            features = torch.empty(
                len(dataset), all_features.shape[-1], device="cuda"
            )
            labels = torch.empty(
                len(dataset), dtype=all_labels.dtype, device="cuda"
            )
        features.index_copy_(0, all_indices, all_features)
        labels.index_copy_(0, all_indices, all_labels)
        if is_main_process():
            print_progress(description, batch_index, len(loader))
    return features, labels


@torch.inference_mode()
def _weighted_knn(
    train_features,
    train_labels,
    test_features,
    test_labels,
    neighbors,
    temperature=0.07,
    num_classes=1000,
):
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
        test_features = test_features[rank::world_size]
        test_labels = test_labels[rank::world_size]
    train_features = train_features.T
    top1 = 0.0
    top5 = 0.0
    total = 0
    images_per_chunk = max(1, test_labels.shape[0] // 100)
    ranges = range(0, test_labels.shape[0], images_per_chunk)
    one_hot = torch.zeros(neighbors, num_classes, device=test_features.device)
    description = f"ImageNet weighted k-NN k={neighbors}"
    total_chunks = len(ranges)
    print(f"{description}: starting {total_chunks} chunks", flush=True)
    for chunk_index, start in enumerate(ranges, start=1):
        feature = test_features[start : start + images_per_chunk]
        target = test_labels[start : start + images_per_chunk]
        similarity = feature @ train_features
        distances, indices = similarity.topk(neighbors, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(target.shape[0], -1)
        retrieved = torch.gather(candidates, 1, indices)
        one_hot.resize_(target.shape[0] * neighbors, num_classes).zero_()
        one_hot.scatter_(1, retrieved.reshape(-1, 1), 1)
        weights = distances.div(temperature).exp()
        probabilities = torch.sum(
            one_hot.view(target.shape[0], neighbors, num_classes)
            * weights.view(target.shape[0], neighbors, 1),
            dim=1,
        )
        predictions = probabilities.argsort(dim=1, descending=True)
        correct = predictions.eq(target.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :5].sum().item()
        total += target.shape[0]
        print_progress(description, chunk_index, total_chunks)
    if dist.is_initialized():
        counts = torch.tensor([top1, top5, total], dtype=torch.float64, device=test_features.device)
        dist.all_reduce(counts)
        top1, top5, total = counts.tolist()
    if total == 0:
        raise ValueError("ImageNet validation set is empty")
    return {
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
    }


def run_imagenet_knn(args, evaluation_name="imagenet_knn"):
    fraction = IMAGENET_KNN_FRACTIONS[evaluation_name]
    percent = round(100 * fraction)
    started = utc_now()
    start_time = time.monotonic()
    model, metadata = load_backbone(
        args.checkpoint, args.checkpoint_key, args.arch
    )
    model.cuda().eval()
    root = _resolve_imagenet_root(args.datasets_root)
    full_train_dataset = datasets.ImageFolder(
        root / "train", transform=_eval_transform()
    )
    if fraction == 1.0:
        train_indices = list(range(len(full_train_dataset)))
        subset_protocol = {
            "training_subset": "all training images in dataset order",
        }
    else:
        train_indices = _simclrv2_subset_indices(full_train_dataset, fraction)
        subset_protocol = _simclrv2_subset_protocol(fraction)
    train_dataset = IndexedSubset(full_train_dataset, train_indices)
    val_dataset = IndexedImageFolder(root / "val", transform=_eval_transform())
    if full_train_dataset.class_to_idx != val_dataset.class_to_idx or len(full_train_dataset.classes) != 1000:
        raise ValueError("ImageNet train/val must share exactly the same 1,000 classes")
    if is_main_process():
        print(
            "ImageNet loaded: "
            f"{len(train_dataset)}/{len(full_train_dataset)} train ({percent}%), "
            f"{len(val_dataset)} val; {subset_protocol['training_subset']}",
            flush=True,
        )
    train_features, train_labels = _extract_distributed_features(
        model, train_dataset, args, "ImageNet train features"
    )
    test_features, test_labels = _extract_distributed_features(
        model, val_dataset, args, "ImageNet val features"
    )
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)
    evaluations = {}
    for neighbors in (10, 20, 100, 200):
        evaluations[str(neighbors)] = _weighted_knn(
            train_features,
            train_labels,
            test_features,
            test_labels,
            neighbors,
        )
    result = None
    if is_main_process():
        result = {
            "evaluation": evaluation_name,
            "dataset": f"ImageNet-1K {percent}%",
            "status": "completed",
            "started_at": started,
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "model": metadata,
            "evaluation_identity": evaluation_identity(args),
            "dataset_sizes": {
                "train": len(train_dataset),
                "full_train": len(full_train_dataset),
                "test": len(val_dataset),
            },
            "protocol": {
                "source": "CRISP Table 4 / original iBOT frozen-feature weighted k-NN",
                "input_resolution": 224,
                "training_fraction": fraction,
                "training_fraction_actual": len(train_dataset) / len(full_train_dataset),
                "training_subset_indices_sha256": _indices_sha256(train_indices),
                "subset_note": (
                    "iBOT uses the predefined SimCLRv2 ImageNet subsets for its "
                    "1% and 10% frozen-feature evaluations."
                    if fraction < 1.0 else "The 100% evaluation uses all ImageNet training images."
                ),
                **subset_protocol,
                "feature": f"final normalized {args.checkpoint_key} CLS token",
                "feature_l2_normalization": True,
                "temperature": 0.07,
                "neighbors": [10, 20, 100, 200],
                "primary_neighbors": 20,
                "gpu_count": dist.get_world_size() if dist.is_initialized() else 1,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            },
            "metrics": evaluations["20"],
            "metrics_by_neighbors": evaluations,
        }
        write_json(args.result_json, result)
        print(
            f"ImageNet {percent}% k-NN (k=20): top-1={result['metrics']['top1']:.3f}, "
            f"top-5={result['metrics']['top5']:.3f}",
            flush=True,
        )
        print(f"Saved results to {args.result_json}", flush=True)
    dist.barrier()
    return result


def imagenet_entrypoint(module, mode, evaluation_name="imagenet_knn"):
    if mode != "knn":
        raise ValueError("This entrypoint supports only ImageNet k-NN")
    percent = round(100 * IMAGENET_KNN_FRACTIONS[evaluation_name])
    parser = base_parser(f"CRISP ImageNet-1K {percent}% k-NN evaluation")
    args = prepare_paths(parser.parse_args(), evaluation_name)
    launch_distributed_if_needed(module)
    initialize_distributed(args.seed, allow_tf32=False)
    try:
        return run_imagenet_knn(args, evaluation_name)
    finally:
        cleanup_distributed()
