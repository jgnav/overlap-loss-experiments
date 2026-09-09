import logging
import tempfile
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from evaluation.utils.common import (
    base_parser,
    cleanup_distributed,
    initialize_distributed,
    launch_distributed_if_needed,
    load_backbone,
    prepare_paths,
    print_progress,
    evaluation_identity,
    utc_now,
    write_json,
)
from evaluation.utils.datasets import DATASET_SPECS, segmentation_manifest


# Compatibility default for callers without a model; production uses the
# checkpoint's patch size through dense_resolution().
DENSE_RESOLUTION = 256


def dense_resolution(patch_size):
    if patch_size not in (14, 16):
        raise ValueError(f"Supported patch sizes are 14 and 16, got {patch_size}")
    return 16 * patch_size


IMAGENET_NORMALIZE = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def _dense_transforms(resolution=DENSE_RESOLUTION):
    image_transform = T.Compose(
        [
            T.Resize(
                (resolution, resolution),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            IMAGENET_NORMALIZE,
        ]
    )

    def target_transform(image):
        image = image.convert("P")
        image = T.functional.resize(
            image,
            (resolution, resolution),
            interpolation=T.InterpolationMode.NEAREST,
        )
        return torch.from_numpy(np.array(image, copy=True))

    return image_transform, target_transform


def _patchify_labels(labels, grid_height, grid_width):
    batch, height, width = labels.shape
    if height % grid_height or width % grid_width:
        raise ValueError(
            f"Target shape {(height, width)} is incompatible with feature grid "
            f"{(grid_height, grid_width)}"
        )
    patch_height = height // grid_height
    patch_width = width // grid_width
    return (
        labels.reshape(
            batch,
            grid_height,
            patch_height,
            grid_width,
            patch_width,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(batch * grid_height * grid_width, patch_height * patch_width)
    )


@torch.inference_mode()
def _extract_features(model, dataset, batch_size, num_workers, description):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    features = None
    labels = None
    offset = 0
    print(f"{description}: starting {len(loader)} batches", flush=True)
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.cuda(non_blocking=True)
        tokens = model.get_intermediate_layers(images, n=1)[0][:, 1:]
        grid_size = math.isqrt(tokens.shape[1])
        if grid_size != 16 or tokens.shape[1] != 256:
            raise ValueError(f"Expected 256 patch tokens, got {tokens.shape[1]}")
        batch_features = tokens.reshape(-1, tokens.shape[-1]).float().cpu()
        batch_labels = _patchify_labels(targets, grid_size, grid_size).cpu()
        if features is None:
            sample_count = len(dataset) * grid_size * grid_size
            features = torch.empty(
                sample_count, batch_features.shape[-1], dtype=torch.float32
            )
            labels = torch.empty(
                sample_count, batch_labels.shape[-1], dtype=batch_labels.dtype
            )
        next_offset = offset + batch_features.shape[0]
        features[offset:next_offset].copy_(batch_features)
        labels[offset:next_offset].copy_(batch_labels)
        offset = next_offset
        print_progress(description, batch_index, len(loader))
    if features is None or labels is None:
        raise ValueError(f"No samples found while extracting {description}")
    if offset != features.shape[0]:
        raise RuntimeError(
            f"Feature extraction stored {offset} of {features.shape[0]} patches"
        )
    return features, labels


def _build_dense_datasets(dataset_name, datasets_root, seed, resolution=DENSE_RESOLUTION):
    spec = DATASET_SPECS[dataset_name]
    image_transform, target_transform = _dense_transforms(resolution)
    full_train = spec["factory"](
        datasets_root,
        spec["train_split"],
        transform=image_transform,
        target_transform=target_transform,
    )
    test = spec["factory"](
        datasets_root,
        spec["test_split"],
        transform=image_transform,
        target_transform=target_transform,
    )
    random_state = np.random.RandomState(seed)
    indices = random_state.permutation(len(full_train)).tolist()
    validation_size = len(full_train) // 10
    validation = Subset(full_train, indices[:validation_size])
    train = Subset(full_train, indices[validation_size:])
    return {"train": train, "val": validation, "test": test}




def _format_capi_result(raw, classifier_name):
    """Translate upstream names into our existing JSON schema; do not rescore."""
    key = "knn" if classifier_name == "knn" else "logreg"
    grid = (
        [{"num_neighbors": k, "distance": distance}
         for k in (1, 3, 10, 30) for distance in ("cosine", "L2")]
        if key == "knn" else
        [{"C": float(c), "max_iter": 1000, "tol": 1e-12,
          "linesearch_max_iter": 50, "lbfgs_hessian_rank": 5}
         for c in 10 ** np.linspace(-6, 5, 8)]
    )
    sweep = []
    for settings in grid:
        suffix = "_".join(f"{name}={value}" for name, value in settings.items())
        score = raw[f"hparam_fitting.{key}.mIoU_{suffix}"]
        sweep.append({**settings, "miou": score, "miou_percent": 100 * score})
    best = max(sweep, key=lambda entry: entry["miou"])
    miou = raw[f"labels_{key}_mIoU"]
    accuracy = raw[f"labels_{key}_acc"]
    return {
        "classifier": "knn" if key == "knn" else "linear_logistic_regression",
        "selected_hyperparameters": {name: best[name] for name in grid[0]},
        "validation_sweep": sweep,
        "metrics": {"miou": miou, "miou_percent": 100 * miou,
                    "pixel_accuracy": accuracy, "pixel_accuracy_percent": 100 * accuracy},
        "capi_raw_metrics": raw,
    }


def run_dense_evaluation(args, dataset_name, classifier_name, evaluation_name):
    from evaluation.vendor.capi.eval_segmentation import eval_model
    from evaluation.utils.capi_adapter import CAPI_REVISION

    started = utc_now()
    start_time = time.monotonic()
    model, metadata = load_backbone(args.checkpoint, args.checkpoint_key, args.arch)
    resolution = dense_resolution(metadata["patch_size"])
    model.cuda().eval()
    spec = DATASET_SPECS[dataset_name]
    full_train = spec["factory"](args.datasets_root, spec["train_split"])
    test = spec["factory"](args.datasets_root, spec["test_split"])
    manifests = {"train": segmentation_manifest(full_train), "test": segmentation_manifest(test)}
    n_train, n_test = len(full_train), len(test)
    identity = evaluation_identity(args)
    if getattr(args, "feature_cache", None) is not None:
        print("Pinned CAPI evaluator extracts fresh features; legacy feature cache is not reused.", flush=True)
    print(
        f"CAPI {CAPI_REVISION}: {dataset_name}, {spec['train_split']} -> "
        f"{spec['test_split']}, resolution={resolution}, patch_size={metadata['patch_size']}, "
        "patch_tokens=256, one GPU", flush=True,
    )
    # Upstream draws its holdout with NumPy's global RNG.
    np.random.seed(args.seed)
    raw = eval_model(
        model,
        train_dataset_name=full_train,
        test_dataset_name=test,
        classifiers=("knn" if classifier_name == "knn" else "logreg",),
        standardization="StandardScaler",
        autocast_dtype=torch.float,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=resolution,
        ignore_labels=spec["ignore_labels"],
        output_dir=str(args.output_dir),
    )
    result = {
        "evaluation": evaluation_name, "dataset": spec["display_name"],
        "status": "completed", "started_at": started, "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start_time,
        "model": metadata, "evaluation_identity": identity,
        "dataset_sizes": {"train": n_train - n_train // 10, "val": n_train // 10, "test": n_test},
        "dataset_manifests": manifests,
        "protocol": {
            "source": "Pinned official CAPI segmentation evaluator with local I/O adapters",
            "capi_revision": CAPI_REVISION,
            "dataset_train_split": spec["train_split"], "dataset_test_split": spec["test_split"],
            "input_resolution": resolution, "patch_tokens": 256,
            "resolution_rule": "16 * checkpoint patch size",
            "backbone_frozen": True,
            "feature": f"final normalized {args.checkpoint_key} patch tokens",
            "standardization": "StandardScaler fitted on train only",
            "validation_split": "seeded 10% of training set",
            "num_classes": spec["num_classes"], "ignore_labels": list(spec["ignore_labels"]),
            "gpu_count": 1,
            "published_score_equivalence": "Exact CRISP dataset lists and CAPI revision not established",
        },
        **_format_capi_result(raw, classifier_name),
    }
    write_json(args.result_json, result)
    print(
        f"{spec['display_name']} {classifier_name}: "
        f"mIoU={result['metrics']['miou_percent']:.3f}, "
        f"accuracy={result['metrics']['pixel_accuracy_percent']:.3f}", flush=True,
    )
    print(f"Saved results to {args.result_json}", flush=True)
    return result


def dense_entrypoint(module, dataset_name, classifier_name, evaluation_name):
    launch_distributed_if_needed(module, required_world_size=1)
    parser = base_parser(
        f"CRISP {DATASET_SPECS[dataset_name]['display_name']} "
        f"{classifier_name} evaluation"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Deprecated compatibility argument; pinned CAPI always extracts fresh features",
    )
    args = prepare_paths(parser.parse_args(), evaluation_name)
    initialize_distributed(args.seed, allow_tf32=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # CAPI's unchanged sweep uses distributed collectives even on one GPU.
    import torch.distributed as dist
    with tempfile.TemporaryDirectory(prefix="capi-rendezvous-") as rendezvous:
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    "nccl", init_method=Path(rendezvous, "store").as_uri(),
                    rank=0, world_size=1,
                )
            return run_dense_evaluation(args, dataset_name, classifier_name, evaluation_name)
        finally:
            cleanup_distributed()
