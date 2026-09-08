import hashlib
import math
import time

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
GPU_COUNT = 4
BATCH_SIZE_PER_GPU = 256
IMAGENET_KNN_TRAINING_FRACTION = 0.10


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


def _stratified_subset_indices(targets, fraction, seed):
    """Select an exact-size, deterministic proportional subset of every class."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Subset fraction must be in (0, 1], got {fraction}")
    targets = torch.as_tensor(targets, dtype=torch.long)
    subset_size = round(len(targets) * fraction)
    classes = torch.unique(targets, sorted=True)
    class_indices = [torch.where(targets == label)[0] for label in classes]
    exact_counts = [len(indices) * fraction for indices in class_indices]
    selected_counts = [math.floor(count) for count in exact_counts]
    remaining = subset_size - sum(selected_counts)
    remainder_order = sorted(
        range(len(classes)),
        key=lambda index: (exact_counts[index] - selected_counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        selected_counts[index] += 1

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for indices, count in zip(class_indices, selected_counts, strict=True):
        order = torch.randperm(len(indices), generator=generator)[:count]
        selected.extend(indices[order].tolist())
    shuffle = torch.randperm(len(selected), generator=generator)
    return [selected[index] for index in shuffle.tolist()]


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
        if is_main_process():
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
    train_features = train_features.T
    top1 = 0.0
    top5 = 0.0
    total = 0
    images_per_chunk = max(1, test_labels.shape[0] // 100)
    ranges = range(0, test_labels.shape[0], images_per_chunk)
    one_hot = torch.zeros(neighbors, num_classes, device="cuda")
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
    return {
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
    }


def run_imagenet_knn(args):
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
    train_indices = _stratified_subset_indices(
        full_train_dataset.targets,
        IMAGENET_KNN_TRAINING_FRACTION,
        args.seed,
    )
    train_dataset = IndexedSubset(full_train_dataset, train_indices)
    val_dataset = IndexedImageFolder(root / "val", transform=_eval_transform())
    if is_main_process():
        print(
            "ImageNet loaded: "
            f"{len(train_dataset)}/{len(full_train_dataset)} train (10%), "
            f"{len(val_dataset)} val; subset seed={args.seed}",
            flush=True,
        )
    train_features, train_labels = _extract_distributed_features(
        model, train_dataset, args, "ImageNet train features"
    )
    test_features, test_labels = _extract_distributed_features(
        model, val_dataset, args, "ImageNet val features"
    )
    result = None
    if is_main_process():
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
        result = {
            "evaluation": "imagenet_knn",
            "dataset": "ImageNet-1K 10%",
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
                "source": "CRISP Section 4.4 / original iBOT weighted k-NN",
                "input_resolution": 224,
                "training_fraction": IMAGENET_KNN_TRAINING_FRACTION,
                "training_subset": "deterministic proportional stratified sample",
                "training_subset_seed": args.seed,
                "training_subset_indices_sha256": _indices_sha256(train_indices),
                "subset_note": (
                    "CRISP specifies a 10% ImageNet-1K k-NN evaluation but does "
                    "not publish subset indices; the seed and index hash make "
                    "this implementation reproducible."
                ),
                "feature": "final normalized teacher CLS token",
                "feature_l2_normalization": True,
                "temperature": 0.07,
                "neighbors": [10, 20, 100, 200],
                "primary_neighbors": 20,
                "gpu_count": GPU_COUNT,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            },
            "metrics": evaluations["20"],
            "metrics_by_neighbors": evaluations,
        }
        write_json(args.result_json, result)
        print(
            f"ImageNet 10% k-NN (k=20): top-1={result['metrics']['top1']:.3f}, "
            f"top-5={result['metrics']['top5']:.3f}",
            flush=True,
        )
        print(f"Saved results to {args.result_json}", flush=True)
    dist.barrier()
    return result


def imagenet_entrypoint(module, mode):
    if mode != "knn":
        raise ValueError("CRISP ablations only use ImageNet-1K 10% k-NN")
    launch_distributed_if_needed(module, required_world_size=GPU_COUNT)
    parser = base_parser("CRISP ImageNet-1K 10% k-NN evaluation")
    args = prepare_paths(parser.parse_args(), f"imagenet_{mode}")
    initialize_distributed(args.seed, allow_tf32=False)
    try:
        return run_imagenet_knn(args)
    finally:
        cleanup_distributed()
