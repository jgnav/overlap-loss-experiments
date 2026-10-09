"""Frozen linear classification using CRISP Table 3/4 and A.2 settings.

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

from evaluation.utils.classification_data import (
    MULTILABEL_DATASETS, VOC_SHOT_EVALUATIONS, few_shot_dataset, make_multilabel_datasets,
)
from evaluation.utils.common import (
    base_parser, cleanup_distributed, evaluation_identity, initialize_distributed,
    launch_distributed_if_needed, load_backbone, prepare_paths, print_progress,
    utc_now, write_json,
)
from evaluation.utils.imagenet import IMAGENET_NORMALIZE, _resolve_imagenet_root


REFERENCE_GPU_COUNT = 4
BATCH_SIZE_PER_GPU = 256
GLOBAL_BATCH_SIZE = REFERENCE_GPU_COUNT * BATCH_SIZE_PER_GPU
BASE_LEARNING_RATE = 0.001


def require_crisp_gpus(world_size):
    if world_size != REFERENCE_GPU_COUNT:
        raise ValueError(
            f"CRISP linear classification requires exactly {REFERENCE_GPU_COUNT} GPUs "
            f"with {BATCH_SIZE_PER_GPU} images per GPU; got {world_size} GPUs"
        )


def classification_transforms(multilabel=False, multilabel_recipe="bce"):
    if multilabel and multilabel_recipe != "ibot":
        # Image-level labels refer to the whole image, including edge objects.
        resize = lambda: T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)
        return (
            T.Compose([resize(), T.RandomHorizontalFlip(), T.ToTensor(), IMAGENET_NORMALIZE]),
            T.Compose([resize(), T.ToTensor(), IMAGENET_NORMALIZE]),
        )
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


def feature_spec(architecture, multilabel=False, multilabel_recipe="bce"):
    if architecture not in ("vit_small", "vit_base", "vit_large"):
        raise ValueError(f"Unsupported linear-probe architecture: {architecture}")
    if multilabel and multilabel_recipe != "ibot":
        return 1, True
    if architecture == "vit_small":
        return 4, False
    if architecture in ("vit_base", "vit_large"):
        return 1, True
    raise ValueError(f"Unsupported linear-probe architecture: {architecture}")


@torch.no_grad()
def classification_features(backbone, images, architecture, multilabel=False, multilabel_recipe="bce"):
    n, average_patches = feature_spec(architecture, multilabel, multilabel_recipe)
    layers = backbone.get_intermediate_layers(images, n=n)
    vectors = [layer[:, 0].float() for layer in layers]
    if average_patches:
        vectors.append(layers[-1][:, 1:].float().mean(dim=1))
    return torch.cat(vectors, dim=-1)


def linear_head(backbone, architecture, num_classes, multilabel=False, multilabel_recipe="bce"):
    n, average_patches = feature_spec(architecture, multilabel, multilabel_recipe)
    head = nn.Linear(backbone.embed_dim * (n + int(average_patches)), num_classes)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)
    return head


def classification_loss(logits, targets, multilabel, loss_name="bce"):
    if not multilabel:
        return F.cross_entropy(logits, targets)
    known = targets >= 0
    truth = targets.clamp_min(0)
    if loss_name == "asymmetric":
        positive = logits.sigmoid()
        negative = (1 - positive + 0.05).clamp(max=1)
        # Match the tested ASL recipe: gamma_pos=0, gamma_neg=4, detached
        # focal weights. Unknown labels are masked after computing elements.
        focal = (1 - (positive * truth + negative * (1 - truth))).pow(4 * (1 - truth)).detach()
        losses = -(truth * positive.clamp_min(1e-8).log()
                   + (1 - truth) * negative.clamp_min(1e-8).log()) * focal
    elif loss_name == "bce":
        losses = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    else:
        raise ValueError(f"Unknown multilabel loss: {loss_name}")
    # Normalize over known labels globally, even if ranks have different numbers
    # of difficult/unknown entries. DDP averages parameter gradients across ranks.
    count = known.sum().to(logits.dtype)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        dist.all_reduce(count)
    if count.item() == 0:
        raise ValueError("Classification batch has no known labels")
    return (losses * known).sum() * world_size / count


def voc2012_average_precision(truth, scores):
    """VOC2010+ precision-envelope AP; inputs contain only known labels.

    Stable score sorting preserves input order for exact ties. AP integrates
    all recall changes, rather than VOC2007's eleven-point approximation.
    """
    positive = np.asarray(truth)[np.argsort(-np.asarray(scores), kind="stable")] == 1
    count = positive.sum()
    if count == 0:
        return 0.0
    tp = np.cumsum(positive)
    recall = tp / count
    precision = tp / np.arange(1, len(positive) + 1)
    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[0.0, precision, 0.0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def multilabel_metrics(targets, scores, classes, dataset_name=None):
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
            ap = (voc2012_average_precision(truth, scores[known, index])
                  if dataset_name == "pascal_voc"
                  else float(average_precision_score(truth, scores[known, index])))
        per_class[name] = ap
    mean_ap = float(np.mean(list(per_class.values())))
    return {
        "map": mean_ap, "map_percent": 100.0 * mean_ap,
        "average_precision_by_class": per_class,
        "classes_without_validation_positives": no_positives,
        "ap_definition": ("VOC2010+ all-points interpolated precision-envelope AP"
                          if dataset_name == "pascal_voc" else "non-interpolated average precision"),
    }


def _make_datasets(args, dataset_name):
    train_transform, val_transform = classification_transforms(
        dataset_name in MULTILABEL_DATASETS, getattr(args, "multilabel_recipe", "bce"),
    )
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


def _protocol(dataset_name, architecture, checkpoint_key, world_size=REFERENCE_GPU_COUNT,
              multilabel_recipe="bce", evaluation_name=None):
    require_crisp_gpus(world_size)
    multilabel = dataset_name != "imagenet"
    epochs = MULTILABEL_DATASETS[dataset_name]["epochs"] if multilabel else 200
    n, average_patches = feature_spec(architecture, multilabel, multilabel_recipe)
    global_batch_size = BATCH_SIZE_PER_GPU * world_size
    protocol = {
        "source": "CRISP Tables 3/4 and Appendix A.2",
        "protocol_version": 4,
        "equivalence": "CRISP's stated resolution, GPU count, batch size and epochs; LR convention and unpublished details remain unverified",
        "protocol_precedence": ["CRISP", "iBOT", "documented local choices"],
        "input_resolution": 224, "gpu_count": world_size,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "global_batch_size": global_batch_size,
        "feature_microbatch_size": BATCH_SIZE_PER_GPU,
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate": BASE_LEARNING_RATE * global_batch_size / 256,
        "learning_rate_scaled_by_batch_size": True,
        "learning_rate_interpretation": "Base LR 0.001 scaled by configured global batch size / 256, following the original iBOT evaluator; CRISP's LR convention remains unpublished",
        "epochs": epochs,
        "backbone_frozen": True, "checkpoint_key": checkpoint_key,
        "classifier": "single linear layer",
        "feature": {"concatenated_cls_blocks": n, "append_mean_patch_tokens": average_patches},
        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 0.0,
        "schedule": "epoch cosine annealing to zero; no warmup",
        "loss": "masked binary cross entropy" if multilabel else "cross entropy",
        "train_transform": "RandomResizedCrop(224, bilinear), horizontal flip, ImageNet normalization",
        "val_transform": "Resize(shorter side=256, bicubic), CenterCrop(224), ImageNet normalization",
        "checkpoint_selection": "final epoch; no selection on reported validation set",
        "training_split": "full training split; VOC low-shot tasks replace training images with per-class random samples",
        "validation_split": "full evaluation split from the dataset manifest" if multilabel else "full official ImageNet-1K validation set",
        "internal_training_holdout": False,
        "setting_sources": {
            "epochs_resolution_gpu_batch_base_lr": "CRISP Appendix A.2; base-LR interpretation uses the original iBOT scaling rule",
            "pooling_optimizer_schedule_transforms_initialization": "Original iBOT linear evaluator; CAPI has no released VOC/COCO/VG multilabel protocol",
            "loss_ap_difficult_labels_split_final_epoch": "Documented local choices; absent from applicable released protocols",
        },
        "metric": "macro average precision (non-interpolated)" if multilabel else "top-1/top-5 accuracy",
        "implementation_choices_not_specified_by_papers": [
            "iBOT architecture-dependent feature pooling, SGD/momentum/weight decay, cosine schedule, augmentation, and head initialization",
            "CRISP's reported LR 0.001 is interpreted as base LR and scaled by global batch size / 256, following the original iBOT evaluator",
            "multilabel masked BCE, unknown-label handling, and non-interpolated macro AP",
            "final-epoch reporting instead of selecting an epoch on the evaluation set",
            "dataset versions, split membership, and label vocabulary are supplied by input manifests",
        ],
    }
    if multilabel and multilabel_recipe != "ibot":
        protocol.update({
            "protocol_version": 5,
            "source": "CRISP multilabel representation and budget; official VOC2012 AP; documented fixed completion choices",
            "equivalence": "Reproducible agreed benchmark protocol; CRISP's unpublished details and split lists are not claimed identical",
            "protocol_precedence": ["CRISP", "official PASCAL VOC", "iBOT optimizer fallback", "documented local choices"],
            "learning_rate": BASE_LEARNING_RATE,
            "learning_rate_scaled_by_batch_size": False,
            "learning_rate_interpretation": "Effective initial LR 0.001, without automatic batch scaling; agreed explicit interpretation of CRISP A.2",
            "feature": {"concatenated_cls_blocks": 1, "append_mean_patch_tokens": True,
                        "normalization": "final backbone LayerNorm", "projection_head": False,
                        "l2_normalization": False, "softmax": False, "standard_scaler": False},
            "loss_reduction": "mean over all known image-class entries across ranks",
            "train_transform": "Resize(entire image to 224x224, bicubic), horizontal flip, ImageNet normalization",
            "val_transform": "Resize(entire image to 224x224, bicubic), ImageNet normalization; no test-time augmentation",
            "metric": ("macro VOC2010+ all-points interpolated precision-envelope AP"
                       if dataset_name == "pascal_voc" else "macro average precision (non-interpolated)"),
            "setting_sources": {
                "epochs_resolution_gpu_batch": "CRISP Table 3 and Appendix A.2",
                "pooling": "Documented local choice: final-LayerNorm CLS concatenated with mean final patch tokens; pooling is not specified in the supplied CRISP PDF",
                "voc_ap_difficult_labels": "Official VOC2012 classification devkit",
                "optimizer_schedule_initialization": "Original iBOT linear evaluator fallback",
                "resize_effective_lr_bce_splits_seed_final_epoch": "Agreed explicit benchmark choices where CRISP is incomplete",
            },
            "implementation_choices_not_specified_by_papers": [
                "whole-image bicubic square resize; horizontal flip only during training",
                "effective initial learning rate 0.001 without batch scaling",
                "final-block, final-LayerNorm CLS and patch tokens before projection; no L2/softmax/scaler",
                "SGD/momentum/weight decay, cosine schedule, linear-head initialization, known-label mean BCE",
                "final-epoch reporting; fixed dataset manifests and seed; few-shot overlap handling",
            ],
        })
        if multilabel_recipe in ("asl224", "asl224_lr001"):
            protocol.update({
                "protocol_version": 6,
                "source": "CRISP stated resolution/budget with the exploratory original-iBOT VOC-selected ASL recipe",
                "equivalence": "Explicit local benchmark recipe; not a recovered CRISP protocol",
                "base_learning_rate": 0.04,
                "learning_rate": 0.04,
                "learning_rate_interpretation": "Effective LR 0.04 selected in the fixed-feature VOC224 sweep; no batch scaling",
                "weight_decay": 0.01,
                "weight_decay_scope": "all linear-head parameters, including bias, matching the sweep",
                "loss": "asymmetric",
                "asymmetric_loss": {"gamma_negative": 4, "gamma_positive": 0,
                                    "probability_clip": 0.05, "detach_focal_weights": True,
                                    "log_epsilon": 1e-8},
                "selection_note": "LR/loss/decay were selected using original-iBOT VOC validation scores; identical settings transferred to both models and all three datasets. Not an independent validation benchmark.",
                "selection_result": "output/analysis/voc_data_protocol_20261008/asl224/results.json",
            })
            protocol["implementation_choices_not_specified_by_papers"] = [
                "whole-image bicubic square resize; horizontal flip only during training",
                "fixed VOC-selected effective LR 0.04, SGD momentum 0.9 and weight/bias decay 0.01",
                "ASL gamma_neg=4, gamma_pos=0, clip=0.05, detached focal weights; globally known-label mean",
                "final-block, final-LayerNorm CLS and patch tokens before projection; no L2/softmax/scaler",
                "final-epoch reporting; fixed dataset manifests and seed; few-shot overlap handling",
            ]
            protocol["setting_sources"]["optimizer_schedule_initialization"] = "VOC224 sweep-selected SGD/decay; iBOT cosine schedule and linear initialization"
            protocol["setting_sources"].pop("resize_effective_lr_bce_splits_seed_final_epoch")
            protocol["setting_sources"]["resize_lr_asl_splits_seed_final_epoch"] = "Explicit local choices; VOC validation sweep selection recorded above"
            if multilabel_recipe == "asl224_lr001":
                protocol.update({
                    "protocol_version": 7,
                    "source": "CRISP stated resolution, budget and effective LR 0.001; retained local ASL/SGD recipe",
                    "base_learning_rate": BASE_LEARNING_RATE,
                    "learning_rate": BASE_LEARNING_RATE,
                    "learning_rate_interpretation": "Effective initial LR 0.001 from CRISP A.2, without batch scaling; user-requested replacement of the exploratory LR 0.04",
                    "selection_note": "LR 0.001 follows the paper as requested. Loss and weight decay retain the earlier VOC-selected settings; unpublished CRISP details remain unverified.",
                })
                protocol["implementation_choices_not_specified_by_papers"][1] = "SGD momentum 0.9 and weight/bias decay 0.01 retained from the earlier VOC-selected recipe"
                protocol["setting_sources"]["learning_rate"] = "CRISP Appendix A.2, interpreted as effective LR 0.001 without batch scaling"
        elif multilabel_recipe != "bce":
            raise ValueError(f"Unknown multilabel recipe: {multilabel_recipe}")
        protocol["multilabel_recipe"] = multilabel_recipe
        shots = VOC_SHOT_EVALUATIONS.get(evaluation_name)
        if shots is not None:
            if dataset_name != "pascal_voc":
                raise ValueError("VOC few-shot evaluation requires PASCAL VOC")
            protocol.update(shots_per_class=shots,
                            validation_split="full original classification validation set")
    if multilabel and multilabel_recipe == "ibot":
        protocol.update({
            "protocol_version": 8,
            "multilabel_recipe": "ibot",
            "source": "Original iBOT ImageNet linear evaluator adapted to multilabel classification",
            "equivalence": "iBOT features, transforms, SGD and LR scaling; local masked BCE/mAP adaptation, CRISP epoch budget and final-epoch reporting",
            "protocol_precedence": ["iBOT linear evaluator", "CRISP epoch budget", "documented multilabel adaptation"],
            "learning_rate_interpretation": "iBOT base LR 0.001 scaled by global batch size / 256",
            "feature": {"concatenated_cls_blocks": n, "append_mean_patch_tokens": average_patches,
                        "normalization": "backbone LayerNorm applied to each selected block",
                        "projection_head": False, "l2_normalization": False,
                        "softmax": False, "standard_scaler": False},
            "loss_reduction": "mean over all known image-class entries across ranks",
            "crop_label_handling": "Image-level labels retained after cropping; no bounding-box relabeling",
            "setting_sources": {
                "features_transforms_optimizer_lr_schedule_initialization": "https://github.com/bytedance/ibot/blob/main/evaluation/eval_linear.py",
                "epochs_gpu_batch": "CRISP Appendix A.2: VOC 500 epochs, COCO/VG 200; 4 GPUs x 256",
                "bce_ap_difficult_labels_splits_seed_final_epoch": "Documented local multilabel adaptation; original iBOT uses multiclass CE and best validation accuracy",
            },
            "implementation_choices_not_specified_by_papers": [
                "masked BCE and non-interpolated macro AP for multilabel classification",
                "final-epoch reporting; fixed dataset manifests and seed; few-shot overlap handling",
                "image-level labels retained after random and center crops",
            ],
        })
        shots = VOC_SHOT_EVALUATIONS.get(evaluation_name)
        if shots is not None:
            if dataset_name != "pascal_voc":
                raise ValueError("VOC few-shot evaluation requires PASCAL VOC")
            protocol.update(shots_per_class=shots,
                            validation_split="full original classification validation set")
    return protocol


def train_epoch(backbone, head, optimizer, loader, architecture, multilabel, device, epoch, rank,
                loss_name="bce", multilabel_recipe="bce"):
    backbone.eval()
    head.train()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for step, (images, targets) in enumerate(loader, start=1):
        images, targets = images.to(device), targets.to(device)
        features = torch.cat([
            classification_features(backbone, chunk, architecture, multilabel, multilabel_recipe)
            for chunk in images.split(BATCH_SIZE_PER_GPU)
        ])
        loss = classification_loss(head(features), targets, multilabel, loss_name)
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
def evaluate(backbone, head, dataset, architecture, multilabel, device, rank, world_size, num_workers, dataset_name=None,
             multilabel_recipe="bce"):
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
        logits = head(classification_features(backbone, images.to(device), architecture, multilabel, multilabel_recipe))
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
        None if multilabel_recipe == "ibot" else dataset_name,
    )


def run_classification(args, dataset_name, evaluation_name, rank, world_size):
    require_crisp_gpus(world_size)
    started, start_time = utc_now(), time.monotonic()
    train, val, dataset_metadata = _make_datasets(args, dataset_name)
    shots = VOC_SHOT_EVALUATIONS.get(evaluation_name)
    if shots is not None:
        if dataset_name != "pascal_voc":
            raise ValueError("Few-shot probes require PASCAL VOC")
        train, sampling_metadata = few_shot_dataset(train, shots, args.seed)
        dataset_metadata = {**dataset_metadata, "few_shot_sampling": sampling_metadata}
    device = torch.device("cuda", torch.cuda.current_device())
    backbone, metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    backbone.to(device).eval()
    architecture = metadata["architecture"]
    multilabel_recipe = getattr(args, "multilabel_recipe", "bce")
    protocol = _protocol(dataset_name, architecture, args.checkpoint_key, world_size,
                         getattr(args, "multilabel_recipe", "bce"), evaluation_name)
    epochs = protocol["epochs"]
    identity = evaluation_identity(args)
    multilabel = dataset_name != "imagenet"
    head = linear_head(backbone, architecture, len(train.classes), multilabel=multilabel,
                       multilabel_recipe=multilabel_recipe).to(device)
    head = nn.parallel.DistributedDataParallel(head, device_ids=[device.index])
    optimizer = torch.optim.SGD(head.parameters(), lr=protocol["learning_rate"],
                                momentum=protocol["momentum"], weight_decay=protocol["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sampler = DistributedSampler(train, shuffle=True, seed=args.seed)
    loader = DataLoader(
        train, sampler=sampler, batch_size=protocol["batch_size_per_gpu"],
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
        loss = train_epoch(backbone, head, optimizer, loader, architecture, multilabel, device, epoch, rank,
                           "asymmetric" if protocol["loss"] == "asymmetric" else "bce", multilabel_recipe)
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
    metrics = evaluate(backbone, head.module, val, architecture, multilabel, device, rank, world_size,
                       args.num_workers, dataset_name, multilabel_recipe)
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
    shots = VOC_SHOT_EVALUATIONS.get(evaluation_name)
    regime = "Full-data" if shots is None else f"{shots}-shot"
    parser = base_parser(f"{regime} {dataset_name} frozen linear classification (CRISP A.2 settings)")
    # Show help before requiring GPUs or starting workers.
    args = prepare_paths(parser.parse_args(), evaluation_name)
    launch_distributed_if_needed(module)
    rank, world_size = initialize_distributed(args.seed, allow_tf32=False)
    try:
        run_classification(args, dataset_name, evaluation_name, rank, world_size)
    finally:
        cleanup_distributed()
