import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
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


# User-selected CRISP A.2 convention: 16 x 16 patch tokens for ViT/16.
DENSE_RESOLUTION = 256


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
        if grid_size * grid_size != tokens.shape[1]:
            raise ValueError(f"Non-square patch-token grid: {tokens.shape[1]}")
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


def _build_dense_datasets(dataset_name, datasets_root, seed):
    spec = DATASET_SPECS[dataset_name]
    image_transform, target_transform = _dense_transforms()
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


def _dataset_metadata(datasets):
    return {
        "train": segmentation_manifest(datasets["train"].dataset),
        "test": segmentation_manifest(datasets["test"]),
    }


def _load_or_extract_features(model, metadata, args, dataset_name):
    datasets = _build_dense_datasets(dataset_name, args.datasets_root, args.seed)
    expected = {
        "dataset": dataset_name,
        "checkpoint_fingerprint": metadata["checkpoint_fingerprint"],
        "checkpoint_key": metadata["checkpoint_key"],
        "architecture": metadata["architecture"],
        "resolution": DENSE_RESOLUTION,
        "seed": args.seed,
        "evaluation_identity": evaluation_identity(args),
        "datasets": _dataset_metadata(datasets),
    }
    if args.feature_cache is not None:
        cache_path = args.feature_cache.expanduser().resolve()
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("metadata") != expected:
                print(f"Recomputing incompatible dense feature cache: {cache_path}", flush=True)
            else:
                print(f"Loading dense features from {cache_path}", flush=True)
                return cached["features"], cached["labels"], datasets

    features = {}
    labels = {}
    for split in ("train", "val", "test"):
        features[split], labels[split] = _extract_features(
            model,
            datasets[split],
            args.batch_size,
            args.num_workers,
            f"{DATASET_SPECS[dataset_name]['display_name']} {split} features",
        )
    if args.feature_cache is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"metadata": expected, "features": features, "labels": labels},
            cache_path,
        )
        print(f"Saved dense features to {cache_path}", flush=True)
    return features, labels, datasets


def _standardize(features):
    print("Fitting CAPI StandardScaler on training features", flush=True)
    scaler = StandardScaler()
    scaler.fit(features["train"].numpy())
    return {
        split: torch.from_numpy(scaler.transform(values.numpy()))
        for split, values in features.items()
    }


def _metric_values(target, prediction, ignore_labels):
    target = target.flatten().cpu().numpy()
    prediction = prediction.flatten().cpu().numpy()
    mask = ~np.isin(target, ignore_labels)
    if not mask.any():
        raise ValueError("All evaluation labels are ignored")
    accuracy = float(np.mean((target[mask] == prediction[mask]).astype(float)))
    miou = float(jaccard_score(target[mask], prediction[mask], average="macro"))
    return {
        "miou": miou,
        "miou_percent": 100.0 * miou,
        "pixel_accuracy": accuracy,
        "pixel_accuracy_percent": 100.0 * accuracy,
    }


class CAPIKNN:
    def __init__(
        self,
        ignore_labels,
        num_neighbors,
        distance,
        inference_batch_size=1024,
        train_chunk_size=262144,
    ):
        self.ignore_labels = tuple(ignore_labels)
        self.num_neighbors = num_neighbors
        self.distance = distance
        self.inference_batch_size = inference_batch_size
        self.train_chunk_size = train_chunk_size

    def fit(self, features, labels):
        majority = labels.mode(dim=-1).values
        keep = ~torch.isin(majority, torch.tensor(self.ignore_labels))
        self.train_features = features[keep].float().cuda(non_blocking=True)
        self.train_labels = labels[keep].cuda(non_blocking=True)

    def clear(self):
        if hasattr(self, "train_features"):
            del self.train_features
            del self.train_labels
        gc.collect()
        torch.cuda.empty_cache()

    def _distance(self, queries, keys):
        if self.distance == "cosine":
            queries = F.normalize(queries, dim=-1)
            keys = F.normalize(keys, dim=-1)
            return 1.0 - queries @ keys.T
        if self.distance == "L2":
            return torch.cdist(queries, keys, p=2)
        raise ValueError(f"Unknown k-NN distance: {self.distance}")

    @torch.compile(dynamic=True)
    def _nearest_in_chunk(self, queries, keys, values):
        distances = self._distance(queries, keys)
        k = min(self.num_neighbors, distances.shape[-1])
        nearest_distances, nearest_indices = torch.topk(
            distances, k, dim=-1, largest=False
        )
        return nearest_distances, values[nearest_indices]

    @torch.inference_mode()
    def predict(self, features, description):
        predictions = torch.empty(
            features.shape[0],
            self.train_labels.shape[-1],
            dtype=self.train_labels.dtype,
        )
        ranges = range(0, features.shape[0], self.inference_batch_size)
        total_batches = len(ranges)
        print(f"{description}: starting {total_batches} batches", flush=True)
        for batch_index, start in enumerate(ranges, start=1):
            queries = features[start : start + self.inference_batch_size].float().cuda()
            chunk_distances = []
            chunk_labels = []
            for chunk_start in range(0, self.train_features.shape[0], self.train_chunk_size):
                keys = self.train_features[
                    chunk_start : chunk_start + self.train_chunk_size
                ]
                values = self.train_labels[
                    chunk_start : chunk_start + self.train_chunk_size
                ]
                nearest_distances, nearest_labels = self._nearest_in_chunk(
                    queries,
                    keys,
                    values,
                )
                chunk_distances.append(nearest_distances.cpu())
                chunk_labels.append(nearest_labels.cpu())
            all_distances = torch.cat(chunk_distances, dim=1)
            all_labels = torch.cat(chunk_labels, dim=1)
            _, selected = torch.topk(
                all_distances,
                min(self.num_neighbors, all_distances.shape[1]),
                dim=1,
                largest=False,
            )
            selected_labels = torch.gather(
                all_labels,
                1,
                selected[..., None].expand(
                    *selected.shape, all_labels.shape[-1]
                ),
            )
            batch_predictions = selected_labels.mode(dim=1).values
            predictions[start : start + len(batch_predictions)] = batch_predictions
            print_progress(description, batch_index, total_batches)
        return predictions


def _evaluate_knn(features, labels, ignore_labels):
    sweep = []
    for num_neighbors in (1, 3, 10, 30):
        for distance in ("cosine", "L2"):
            classifier = CAPIKNN(ignore_labels, num_neighbors, distance)
            classifier.fit(features["train"], labels["train"])
            prediction = classifier.predict(
                features["val"],
                f"CAPI k-NN val k={num_neighbors} {distance}",
            )
            metrics = _metric_values(labels["val"], prediction, ignore_labels)
            sweep.append(
                {
                    "num_neighbors": num_neighbors,
                    "distance": distance,
                    **metrics,
                }
            )
            print(
                f"k={num_neighbors} distance={distance}: "
                f"val mIoU={metrics['miou_percent']:.3f}",
                flush=True,
            )
            classifier.clear()
    best = max(sweep, key=lambda item: item["miou"])
    classifier = CAPIKNN(
        ignore_labels, best["num_neighbors"], best["distance"]
    )
    classifier.fit(
        torch.cat((features["train"], features["val"])),
        torch.cat((labels["train"], labels["val"])),
    )
    prediction = classifier.predict(features["test"], "CAPI k-NN test")
    test_metrics = _metric_values(labels["test"], prediction, ignore_labels)
    classifier.clear()
    return {
        "classifier": "knn",
        "selected_hyperparameters": {
            "num_neighbors": best["num_neighbors"],
            "distance": best["distance"],
        },
        "validation_sweep": sweep,
        "metrics": test_metrics,
    }


class CAPILogisticRegression:
    def __init__(self, ignore_labels, regularization):
        try:
            import cuml.linear_model
        except ImportError as error:
            raise ImportError(
                "The exact CAPI linear protocol requires RAPIDS cuML. Install "
                "the repository requirements.txt in the evaluation environment."
            ) from error
        self.ignore_labels = tuple(ignore_labels)
        self.estimator = cuml.linear_model.LogisticRegression(
            penalty="l2",
            C=regularization,
            max_iter=1000,
            output_type="numpy",
            tol=1e-12,
            linesearch_max_iter=50,
            verbose=False,
        )
        self.estimator.solver_model.lbfgs_memory = 5

    def fit(self, features, labels):
        majority = labels.mode(dim=-1).values
        keep = ~torch.isin(majority, torch.tensor(self.ignore_labels))
        self.patch_pixels = labels.shape[-1]
        self.estimator.fit(
            features[keep].numpy(), majority[keep].flatten().numpy()
        )

    def predict(self, features, description):
        prediction = torch.empty(
            features.shape[0], self.patch_pixels, dtype=torch.uint8
        )
        batch_size = 1024
        ranges = range(0, features.shape[0], batch_size)
        total_batches = len(ranges)
        print(f"{description}: starting {total_batches} batches", flush=True)
        for batch_index, start in enumerate(ranges, start=1):
            labels = torch.from_numpy(
                self.estimator.predict(features[start : start + batch_size].numpy())
            ).to(torch.uint8)
            prediction[start : start + len(labels)] = labels[:, None].expand(
                -1, self.patch_pixels
            )
            print_progress(description, batch_index, total_batches)
        return prediction


def _evaluate_linear(features, labels, ignore_labels):
    regularizations = tuple(float(value) for value in 10 ** np.linspace(-6, 5, 8))
    sweep = []
    for regularization in regularizations:
        print(
            f"Fitting CAPI linear probe with C={regularization:.6g}",
            flush=True,
        )
        classifier = CAPILogisticRegression(ignore_labels, regularization)
        classifier.fit(features["train"], labels["train"])
        prediction = classifier.predict(
            features["val"], f"CAPI linear val C={regularization:.3g}"
        )
        metrics = _metric_values(labels["val"], prediction, ignore_labels)
        sweep.append({"C": regularization, **metrics})
        print(
            f"C={regularization:.6g}: val mIoU={metrics['miou_percent']:.3f}",
            flush=True,
        )
        del classifier
        gc.collect()
        torch.cuda.empty_cache()
    best = max(sweep, key=lambda item: item["miou"])
    print(
        f"Refitting CAPI linear probe on train+val with C={best['C']:.6g}",
        flush=True,
    )
    classifier = CAPILogisticRegression(ignore_labels, best["C"])
    classifier.fit(
        torch.cat((features["train"], features["val"])),
        torch.cat((labels["train"], labels["val"])),
    )
    prediction = classifier.predict(features["test"], "CAPI linear test")
    test_metrics = _metric_values(labels["test"], prediction, ignore_labels)
    return {
        "classifier": "linear_logistic_regression",
        "selected_hyperparameters": {
            "C": best["C"],
            "solver": "cuml L-BFGS",
            "max_iter": 1000,
            "tol": 1e-12,
            "linesearch_max_iter": 50,
            "lbfgs_hessian_rank": 5,
        },
        "validation_sweep": sweep,
        "metrics": test_metrics,
    }


def run_dense_evaluation(args, dataset_name, classifier_name, evaluation_name):
    started = utc_now()
    start_time = time.monotonic()
    model, metadata = load_backbone(
        args.checkpoint, args.checkpoint_key, args.arch
    )
    model.cuda().eval()
    features, labels, datasets = _load_or_extract_features(
        model, metadata, args, dataset_name
    )
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    features = _standardize(features)
    spec = DATASET_SPECS[dataset_name]
    if classifier_name == "knn":
        evaluation = _evaluate_knn(features, labels, spec["ignore_labels"])
    elif classifier_name == "linear":
        evaluation = _evaluate_linear(features, labels, spec["ignore_labels"])
    else:
        raise ValueError(f"Unknown dense classifier: {classifier_name}")
    result = {
        "evaluation": evaluation_name,
        "dataset": spec["display_name"],
        "status": "completed",
        "started_at": started,
        "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - start_time,
        "model": metadata,
        "evaluation_identity": evaluation_identity(args),
        "dataset_sizes": {key: len(value) for key, value in datasets.items()},
        "dataset_manifests": _dataset_metadata(datasets),
        "protocol": {
            "source": "CRISP Appendix A.2 / official CAPI segmentation evaluation",
            "input_resolution": DENSE_RESOLUTION,
            "patch_tokens": (DENSE_RESOLUTION // metadata["patch_size"]) ** 2,
            "backbone_frozen": True,
            "feature": f"final normalized {args.checkpoint_key} patch tokens",
            "standardization": "StandardScaler fitted on train only",
            "validation_split": "seeded 10% of training set",
            "num_classes": spec["num_classes"],
            "ignore_labels": list(spec["ignore_labels"]),
            "gpu_count": 1,
        },
        **evaluation,
    }
    write_json(args.result_json, result)
    print(
        f"{spec['display_name']} {classifier_name}: "
        f"mIoU={result['metrics']['miou_percent']:.3f}, "
        f"accuracy={result['metrics']['pixel_accuracy_percent']:.3f}",
        flush=True,
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
        help="Optional reusable train/val/test patch-feature cache",
    )
    args = prepare_paths(parser.parse_args(), evaluation_name)
    initialize_distributed(args.seed, allow_tf32=True)
    try:
        return run_dense_evaluation(
            args, dataset_name, classifier_name, evaluation_name
        )
    finally:
        cleanup_distributed()
