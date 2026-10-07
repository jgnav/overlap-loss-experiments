"""Legacy CAPI classification comparison, excluded from the CRISP linear suite.

Retained for prior-result interpretation and CAPI adapter tests. The configured
imagenet_linear evaluation now uses evaluation.utils.classification.
"""

import copy
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torchvision.datasets import ImageFolder

from evaluation.utils.capi_adapter import CAPI_REVISION
from evaluation.utils.common import (
    base_parser, cleanup_distributed, evaluation_identity, initialize_distributed,
    launch_distributed_if_needed, load_backbone, prepare_paths, utc_now, write_json,
)
from evaluation.utils.imagenet import _resolve_imagenet_root

EPOCHS = 200
WARMUP_ITERATIONS = 1_250
GLOBAL_BATCH_SIZE = 1_024
LEARNING_RATES = (0.001,)
WEIGHT_DECAYS = (5e-4, 1e-3, 5e-2)


def make_dataset(dataset_str_or_path, transform=None, target_transform=None, cache_policy=None):
    """Clone the supplied ImageFolder; train and holdout need separate transforms."""
    if not isinstance(dataset_str_or_path, ImageFolder):
        raise TypeError("CAPI classification adapter expects a local ImageFolder")
    dataset = copy.copy(dataset_str_or_path)
    dataset.transform, dataset.target_transform = transform, target_transform
    return dataset


class CAPIBackbone(nn.Module):
    """Expose final CLS, register and spatial patch tokens in CAPI's model API."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_embed.patch_size

    def forward(self, images):
        tokens = self.backbone.get_intermediate_layers(images, n=1)[0]
        patches = tokens[:, 1:]
        height, width = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        if patches.shape[1] != height * width:
            raise ValueError("Backbone patch grid does not match the input")
        # The intermediate-layer API omits registers; CAPI's default probes
        # use CLS/patches only, so an empty register tensor is sufficient.
        return tokens[:, 0], tokens[:, :0], patches.reshape(len(images), height, width, -1)


def protocol(world_size, training_samples=None):
    if GLOBAL_BATCH_SIZE % world_size:
        raise ValueError("CAPI requires a GPU count dividing global batch size 1024")
    return {
        "source": "CRISP Appendix A.2 settings; pinned CAPI classification fallback",
        "protocol_precedence": ["CRISP", "CAPI", "iBOT"],
        "capi_revision": CAPI_REVISION,
        "epochs": EPOCHS,
        "iterations": math.ceil(EPOCHS * training_samples / GLOBAL_BATCH_SIZE) if training_samples is not None else None,
        "iteration_rule": "ceil(200 * actual training-split images / global batch); CAPI infinite sampler",
        "warmup_iterations": WARMUP_ITERATIONS,
        "global_batch_size": GLOBAL_BATCH_SIZE, "batch_size_per_gpu": GLOBAL_BATCH_SIZE // world_size,
        "gpu_count": world_size, "input_resolution": 224, "backbone_frozen": True,
        "optimizer": "AdamW", "betas": [0.9, 0.95],
        "learning_rates": list(LEARNING_RATES), "weight_decays": list(WEIGHT_DECAYS),
        "learning_rate_scaling": "base learning rate * global batch size / 256",
        "actual_initial_learning_rates": [rate * GLOBAL_BATCH_SIZE / 256 for rate in LEARNING_RATES],
        "learning_rate_interpretation": "CRISP 0.001 treated as base LR using CAPI/iBOT scaling; author convention unverified",
        "bias_weight_decay": 0.0, "schedule": "linear warmup then cosine decay to zero",
        "representations": ["cls", "avg_patch", "cls_avg_patch", "patch"],
        "feature": "final normalized CLS/patch tokens; no concatenation of intermediate blocks",
        "validation_split": "10% of ImageNet train; numpy.default_rng(42)",
        "sampler": "pinned CAPI InfiniteSampler, seed 42",
        "hyperparameter_selection": "best heldout top-1 separately for each feature source",
        "primary_metric": "top-1 of the selected CLS linear classifier",
        "attentive_metric": "top-1 of the selected patch attentive classifier",
        "train_transform": "RandomResizedCrop(224, bicubic), horizontal flip, ImageNet normalization",
        "test_transform": "Resize(256, bicubic), CenterCrop(224), ImageNet normalization",
        "test_split": "official ImageNet validation; never used for hyperparameter selection",
        "checkpoint_period": 1250, "validation_period": 1250,
        "use_compile": False, "compile_note": "execution optimization only; eager execution",
        "setting_sources": {
            "epochs_resolution_gpu_batch_base_lr": "CRISP Appendix A.2",
            "features_optimizer_warmup_transforms_holdout_weight_decay_selection": "Pinned CAPI evaluator",
            "lr_scaling": "CAPI/iBOT fallback assumption",
        },
        "published_score_equivalence": "CRISP stated settings with CAPI fallback details; exact author recipe unverified",
    }


def split_metadata(dataset):
    order = hashlib.sha256()
    for image, target in dataset.samples:
        order.update(f"{Path(image).relative_to(dataset.root)}\t{target}\n".encode())
    indices = np.random.default_rng(42).permutation(len(dataset))
    train_count = round(0.9 * len(dataset))
    return {
        "samples": len(dataset), "ordered_samples_sha256": order.hexdigest(),
        "train": train_count, "holdout": len(dataset) - train_count,
        "training_indices_sha256": hashlib.sha256(indices[:train_count].tobytes()).hexdigest(),
        "holdout_indices_sha256": hashlib.sha256(indices[train_count:].tobytes()).hexdigest(),
    }


def run(args, rank, world_size):
    from evaluation.vendor.capi.eval_classification import eval_model

    started, timer = utc_now(), time.monotonic()
    root = _resolve_imagenet_root(args.datasets_root)
    train, test = ImageFolder(root / "train"), ImageFolder(root / "val")
    if train.class_to_idx != test.class_to_idx or len(train.classes) != 1000:
        raise ValueError("ImageNet train/val must share exactly 1000 classes")
    backbone, model_metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    dataset_metadata = split_metadata(train)
    recipe = protocol(world_size, dataset_metadata["train"])
    signature = {"evaluation_identity": evaluation_identity(args), "model": model_metadata,
                 "protocol": recipe, "dataset": dataset_metadata}
    signature_path = args.output_dir / "protocol.json"
    if signature_path.is_file() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Existing CAPI probe belongs to a different checkpoint/protocol; use a new output directory")
    if rank == 0:
        write_json(signature_path, signature)
    dist.barrier()
    raw = eval_model(
        CAPIBackbone(backbone).cuda().eval(), metric_dumper=lambda _: None,
        output_dir=str(args.output_dir), train_dataset_name=train,
        test_dataset_names=(test,), val_proportion=0.1,
        representations=("cls", "avg_patch", "patch"),
        n_iters=recipe["iterations"], warmup_iters=WARMUP_ITERATIONS,
        learning_rates=LEARNING_RATES, weight_decays=WEIGHT_DECAYS,
        batch_size=recipe["batch_size_per_gpu"], num_classes=1000,
        num_workers=args.num_workers, use_compile=False, dataset_use_cache=False,
    )
    if rank == 0:
        rows = json.loads((args.output_dir / "test_classifiers.json").read_text())
        features = {row["feature_source"]: {
            "top1": 100.0 * row["acc"],
            "selected_learning_rate": row["classifier_params"][0],
            "selected_weight_decay": row["classifier_params"][1],
        } for row in rows}
        result = {
            "evaluation": "imagenet_linear", "task": "multiclass_classification",
            "dataset": "ImageNet-1K", "status": "completed",
            "started_at": started, "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - timer,
            "model": model_metadata, "evaluation_identity": signature["evaluation_identity"],
            "dataset_sizes": {"train": signature["dataset"]["train"], "val": signature["dataset"]["holdout"], "test": len(test)},
            "dataset_manifest": signature["dataset"], "protocol": recipe,
            "metrics": {"top1": features["cls"]["top1"], "attentive_top1": features["patch"]["top1"]},
            "feature_results": features, "capi_raw_metrics": raw,
            "validation_sweep": json.loads((args.output_dir / "validation_sweep.json").read_text()),
        }
        write_json(args.result_json, result)
        print(f"CAPI ImageNet: CLS linear={result['metrics']['top1']:.3f}, patch attentive={result['metrics']['attentive_top1']:.3f}", flush=True)
    dist.barrier()


def entrypoint():
    args = prepare_paths(base_parser("Pinned CAPI ImageNet linear/attentive evaluation").parse_args(), "imagenet_linear")
    launch_distributed_if_needed("evaluation.utils.imagenet_linear")
    rank, world_size = initialize_distributed(args.seed, allow_tf32=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        run(args, rank, world_size)
    finally:
        cleanup_distributed()
