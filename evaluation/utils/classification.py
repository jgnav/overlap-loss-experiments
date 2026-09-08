"""Full-data linear classification using the settings stated in CRISP A.2.

The papers do not specify a complete recipe. iBOT-derived implementation choices
are identified in the saved protocol metadata and evaluation/README.md.
"""

import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms as T

from evaluation.utils.classification_data import MULTILABEL_DATASETS, make_multilabel_datasets
from evaluation.utils.common import (
    base_parser, cleanup_distributed, evaluation_identity, initialize_distributed,
    launch_distributed_if_needed, load_backbone, prepare_paths, print_progress,
    utc_now, write_json,
)
from evaluation.utils.imagenet import IMAGENET_NORMALIZE, _resolve_imagenet_root


GPU_COUNT = 4
BATCH_SIZE_PER_GPU = 256
LEARNING_RATE = 0.001


def classification_transforms():
    # Original iBOT linear-probe transforms (training RRC uses bilinear).
    train = T.Compose([
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), IMAGENET_NORMALIZE,
    ])
    val = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
        T.ToTensor(), IMAGENET_NORMALIZE,
    ])
    return train, val


def feature_spec(architecture):
    if architecture == "vit_small":
        return 4, False
    if architecture in ("vit_base", "vit_large"):
        return 1, True
    raise ValueError(f"Unsupported linear-probe architecture: {architecture}")


@torch.no_grad()
def classification_features(backbone, images, architecture):
    n, average_patches = feature_spec(architecture)
    layers = backbone.get_intermediate_layers(images, n=n)
    vectors = [layer[:, 0].float() for layer in layers]
    if average_patches:
        vectors.append(layers[-1][:, 1:].float().mean(dim=1))
    return torch.cat(vectors, dim=-1)


def linear_head(backbone, architecture, num_classes):
    n, average_patches = feature_spec(architecture)
    head = nn.Linear(backbone.embed_dim * (n + int(average_patches)), num_classes)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)
    return head


def classification_loss(logits, targets, multilabel):
    if not multilabel:
        return F.cross_entropy(logits, targets)
    known = targets >= 0
    losses = F.binary_cross_entropy_with_logits(logits, targets.clamp_min(0), reduction="none")
    # Normalize over known labels globally, even if ranks have different numbers
    # of difficult/unknown entries. DDP averages parameter gradients across ranks.
    count = known.sum().to(logits.dtype)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        dist.all_reduce(count)
    if count.item() == 0:
        raise ValueError("Classification batch has no known labels")
    return (losses * known).sum() * world_size / count


def multilabel_metrics(targets, scores, classes):
    targets, scores = np.asarray(targets), np.asarray(scores)
    if targets.shape != scores.shape or targets.ndim != 2 or targets.shape[1] != len(classes):
        raise ValueError("Targets, predictions, and class vocabulary have incompatible shapes")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite classification scores")
    per_class = {}
    no_positives = []
    for index, name in enumerate(classes):
        known = targets[:, index] >= 0
        truth = targets[known, index]
        if not known.any():
            raise ValueError(f"No known validation labels for class {name}")
        if not (truth == 1).any():
            # Keep the fixed vocabulary in the denominator; never silently
            # improve mAP by dropping a class with no positives.
            ap = 0.0
            no_positives.append(name)
        else:
            ap = float(average_precision_score(truth, scores[known, index]))
        per_class[name] = ap
    mean_ap = float(np.mean(list(per_class.values())))
    return {
        "map": mean_ap, "map_percent": 100.0 * mean_ap,
        "average_precision_by_class": per_class,
        "classes_without_validation_positives": no_positives,
    }


def _make_datasets(args, dataset_name):
    train_transform, val_transform = classification_transforms()
    if dataset_name in MULTILABEL_DATASETS:
        return make_multilabel_datasets(args, dataset_name, train_transform, val_transform)
    if dataset_name != "imagenet":
        raise ValueError(f"Unknown classification dataset: {dataset_name}")
    root = _resolve_imagenet_root(args.datasets_root)
    train = datasets.ImageFolder(root / "train", transform=train_transform)
    val = datasets.ImageFolder(root / "val", transform=val_transform)
    if train.class_to_idx != val.class_to_idx or len(train.classes) != 1000:
        raise ValueError("ImageNet train/val must share exactly the same 1,000 classes")
    return train, val, {"root": str(root), "classes": train.classes, "training_fraction": 1.0}


def _protocol(dataset_name, architecture, checkpoint_key):
    n, average_patches = feature_spec(architecture)
    return {
        "source": "CRISP Appendix A.2; CG-SSL Table 2 task coverage",
        "equivalence": "Matches stated CRISP settings; unpublished choices remain unverified",
        "input_resolution": 224, "gpu_count": GPU_COUNT,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "learning_rate": LEARNING_RATE,
        "learning_rate_scaled_by_batch_size": False,
        "epochs": MULTILABEL_DATASETS.get(dataset_name, {"epochs": 200})["epochs"],
        "backbone_frozen": True, "checkpoint_key": checkpoint_key,
        "feature": {"concatenated_cls_blocks": n, "append_mean_patch_tokens": average_patches},
        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 0.0,
        "schedule": "epoch cosine annealing to zero; no warmup",
        "loss": "masked binary cross entropy" if dataset_name != "imagenet" else "cross entropy",
        "train_transform": "RandomResizedCrop(224, bilinear), horizontal flip, ImageNet normalization",
        "val_transform": "Resize(shorter side=256, bicubic), CenterCrop(224), ImageNet normalization",
        "checkpoint_selection": "final epoch; no selection on reported validation set",
        "metric": "macro average precision (non-interpolated)" if dataset_name != "imagenet" else "top-1/top-5 accuracy",
        "implementation_choices_not_specified_by_papers": [
            "iBOT architecture-dependent feature pooling, SGD/momentum/weight decay, cosine schedule, augmentation, and head initialization",
            "0.001 is interpreted as the actual optimizer learning rate, without iBOT's batch-size rescaling",
            "multilabel masked BCE, unknown-label handling, and non-interpolated macro AP",
            "final-epoch reporting instead of selecting an epoch on the evaluation set",
            "dataset versions, split membership, and label vocabulary are supplied by input manifests",
        ],
    }


def train_epoch(backbone, head, optimizer, loader, architecture, multilabel, device, epoch, rank):
    backbone.eval()
    head.train()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for step, (images, targets) in enumerate(loader, start=1):
        images, targets = images.to(device), targets.to(device)
        features = classification_features(backbone, images, architecture)
        loss = classification_loss(head(features), targets, multilabel)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite probe loss at epoch {epoch + 1}, batch {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total[0] += loss.detach().double() * len(images)
        total[1] += len(images)
        if rank == 0:
            print_progress(f"Linear probe epoch {epoch + 1}", step, len(loader))
    if dist.is_initialized():
        dist.all_reduce(total)
    return float((total[0] / total[1]).item())


@torch.no_grad()
def evaluate(backbone, head, dataset, architecture, multilabel, device, rank, world_size, num_workers):
    # Unequal rank lengths are intentional. Using the unwrapped head avoids DDP
    # forward collectives; each validation image contributes exactly once.
    head.eval()
    loader = DataLoader(
        Subset(dataset, range(rank, len(dataset), world_size)),
        batch_size=BATCH_SIZE_PER_GPU, num_workers=num_workers, pin_memory=True,
    )
    predictions, labels = [], []
    counts = torch.zeros(3, dtype=torch.float64, device=device)
    for step, (images, targets) in enumerate(loader, start=1):
        logits = head(classification_features(backbone, images.to(device), architecture))
        if multilabel:
            predictions.append(logits.cpu())
            labels.append(targets.cpu())
        else:
            correct = logits.topk(5, dim=1).indices.eq(targets.to(device)[:, None])
            counts[0] += correct[:, 0].sum()
            counts[1] += correct.any(dim=1).sum()
            counts[2] += len(targets)
        if rank == 0:
            print_progress("Linear probe validation", step, len(loader))
    if not multilabel:
        if dist.is_initialized():
            dist.all_reduce(counts)
        return {"top1": float(100 * counts[0] / counts[2]), "top5": float(100 * counts[1] / counts[2])}
    width = len(dataset.classes)
    local = (
        torch.cat(labels).numpy() if labels else np.empty((0, width)),
        torch.cat(predictions).numpy() if predictions else np.empty((0, width)),
    )
    gathered = [None] * world_size if rank == 0 else None
    if dist.is_initialized():
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    if rank != 0:
        return None
    return multilabel_metrics(
        np.concatenate([item[0] for item in gathered]),
        np.concatenate([item[1] for item in gathered]), dataset.classes,
    )


def run_classification(args, dataset_name, evaluation_name, rank, world_size):
    started, start_time = utc_now(), time.monotonic()
    train, val, dataset_metadata = _make_datasets(args, dataset_name)
    device = torch.device("cuda", torch.cuda.current_device())
    backbone, metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    backbone.to(device).eval()
    architecture = metadata["architecture"]
    protocol = _protocol(dataset_name, architecture, args.checkpoint_key)
    epochs = protocol["epochs"]
    identity = evaluation_identity(args)
    multilabel = dataset_name != "imagenet"
    head = linear_head(backbone, architecture, len(train.classes)).to(device)
    head = nn.parallel.DistributedDataParallel(head, device_ids=[device.index])
    optimizer = torch.optim.SGD(head.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sampler = DistributedSampler(train, shuffle=True, seed=args.seed)
    loader = DataLoader(
        train, sampler=sampler, batch_size=BATCH_SIZE_PER_GPU,
        num_workers=args.num_workers, pin_memory=True,
        # Per-epoch worker reseeding makes resumed augmentations reproducible.
        persistent_workers=False,
    )
    checkpoint_path = args.output_dir / "linear_checkpoint.pth"
    signature = {"evaluation_identity": identity, "model": metadata, "dataset": dataset_metadata, "protocol": protocol}
    first_epoch = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError(f"Probe checkpoint does not match this evaluation: {checkpoint_path}")
        head.module.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first_epoch = saved["epoch"]
    if rank == 0:
        write_json(args.output_dir / "protocol.json", {**signature, "dataset_sizes": {"train": len(train), "val": len(val)}})
        print(f"Starting {evaluation_name}: {epochs} epochs, {len(train)} train, {len(val)} val", flush=True)
    for epoch in range(first_epoch, epochs):
        # Each rank gets different, but restart-stable, stochastic augmentations.
        torch.manual_seed(args.seed + epoch * world_size + rank)
        sampler.set_epoch(epoch)
        loss = train_epoch(backbone, head, optimizer, loader, architecture, multilabel, device, epoch, rank)
        scheduler.step()
        if rank == 0:
            write_json(args.output_dir / "progress.json", {"epoch": epoch + 1, "epochs": epochs, "train_loss": loss})
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({
                "signature": signature, "epoch": epoch + 1,
                "head": head.module.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, temporary)
            temporary.replace(checkpoint_path)
        dist.barrier()
    metrics = evaluate(backbone, head.module, val, architecture, multilabel, device, rank, world_size, args.num_workers)
    if rank == 0:
        result = {
            "evaluation": evaluation_name,
            "task": "multilabel_classification" if multilabel else "multiclass_classification",
            "dataset": MULTILABEL_DATASETS[dataset_name]["display_name"] if multilabel else "ImageNet-1K",
            "status": "completed", "started_at": started, "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "model": metadata, "evaluation_identity": identity,
            "dataset_sizes": {"train": len(train), "test": len(val)},
            "dataset_manifest": dataset_metadata, "protocol": protocol, "metrics": metrics,
        }
        write_json(args.result_json, result)
        primary = metrics["map_percent"] if multilabel else metrics["top1"]
        print(f"Completed {evaluation_name}: {'mAP' if multilabel else 'top-1'}={primary:.3f}", flush=True)
    dist.barrier()


def classification_entrypoint(module, dataset_name, evaluation_name):
    parser = base_parser(f"Full-data {dataset_name} frozen linear classification (CRISP A.2 settings)")
    # Show help before requiring GPUs or starting four processes.
    args = prepare_paths(parser.parse_args(), evaluation_name)
    launch_distributed_if_needed(module, required_world_size=GPU_COUNT)
    rank, world_size = initialize_distributed(args.seed, allow_tf32=False)
    try:
        run_classification(args, dataset_name, evaluation_name, rank, world_size)
    finally:
        cleanup_distributed()
