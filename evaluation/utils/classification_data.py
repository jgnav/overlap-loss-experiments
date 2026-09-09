"""Explicit split and label inputs for full-data multilabel linear probing."""

import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MULTILABEL_DATASETS = {
    "pascal_voc": {"display_name": "PASCAL VOC", "num_classes": 20, "epochs": 500},
    "coco": {"display_name": "MS-COCO", "num_classes": 80, "epochs": 200},
}


class MultilabelDataset(Dataset):
    def __init__(self, samples, classes, transform=None):
        self.images = [sample[0] for sample in samples]
        self.targets = torch.from_numpy(np.asarray([sample[1] for sample in samples], dtype=np.float32))
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def read_multilabel_manifest(path, datasets_root, dataset_name, num_classes):
    """Read explicit dataset versions, image splits, and class vocabularies.

    Labels have one unambiguous convention: 1 positive, 0 negative, null ignored.
    In particular raw VOC -1/0/1 labels must be converted by the data preparer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing multilabel manifest: {path}. See evaluation/README.md for "
            "the required train/val images and ordered class-label schema."
        )
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("dataset") != dataset_name:
        raise ValueError(f"{path}: dataset must be {dataset_name!r}")
    if not isinstance(manifest.get("source"), str) or not manifest["source"].strip():
        raise ValueError(f"{path}: source must identify the split/annotation provenance")
    classes = manifest.get("classes")
    if (not isinstance(classes, list) or len(classes) != num_classes
            or any(not isinstance(name, str) or not name.strip() for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError(f"{path}: classes must contain {num_classes} unique nonempty names")
    splits = manifest.get("splits", {})
    if set(splits) != {"train", "val"}:
        raise ValueError(f"{path}: splits must contain exactly train and val")
    samples = {}
    seen_ids, seen_paths = set(), set()
    resolved_parents = {}
    for split in ("train", "val"):
        rows = splits[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: {split} must be a nonempty list")
        samples[split] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("image"), str):
                raise ValueError(f"{path}: each sample needs an image path")
            image = Path(row["image"])
            if not image.is_absolute():
                image = Path(datasets_root) / image
            image = image.expanduser()
            # Large manifests share very few image directories. Resolve those
            # directories once, not all their ancestors for every image. Keep
            # resolving file symlinks so aliases cannot bypass overlap checks.
            parent = image.parent
            if parent not in resolved_parents:
                resolved_parents[parent] = parent.resolve()
            image = resolved_parents[parent] / image.name
            try:
                mode = image.lstat().st_mode
            except (FileNotFoundError, NotADirectoryError):
                raise FileNotFoundError(f"Manifest image does not exist: {image}") from None
            if stat.S_ISLNK(mode):
                image = image.resolve()
                is_file = image.is_file()
            else:
                is_file = stat.S_ISREG(mode)
            image_id = row.get("id", str(image))
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"{path}: image IDs must be nonempty strings")
            if image_id in seen_ids or image in seen_paths:
                raise ValueError(f"{path}: duplicate image or train/val overlap: {image_id}")
            seen_ids.add(image_id)
            seen_paths.add(image)
            if not is_file:
                raise FileNotFoundError(f"Manifest image does not exist: {image}")
            labels = row.get("labels")
            if (not isinstance(labels, list) or len(labels) != num_classes
                    or any(value is not None and (type(value) is not int or value not in (0, 1))
                           for value in labels)):
                raise ValueError(f"{path}: labels must have {num_classes} entries from 0, 1, null")
            if all(value is None for value in labels):
                raise ValueError(f"{path}: sample has no known labels: {image_id}")
            samples[split].append((image, [-1 if value is None else value for value in labels]))
    targets = np.asarray([row[1] for row in samples["train"]])
    if np.any((targets == 1).sum(0) == 0) or np.any((targets == 0).sum(0) == 0):
        raise ValueError(f"{path}: full-data training needs positive and negative examples for every class")
    val_targets = np.asarray([row[1] for row in samples["val"]])
    if np.any((val_targets >= 0).sum(0) == 0):
        raise ValueError(f"{path}: validation needs known labels for every class")
    metadata = {
        "manifest": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "annotation_source": manifest["source"],
        "classes": classes,
        "label_encoding": "0=negative, 1=positive, null=ignored",
        "split_sizes": {split: len(rows) for split, rows in samples.items()},
        "validation_positives_per_class": (val_targets == 1).sum(0).tolist(),
    }
    return samples, classes, metadata


def make_multilabel_datasets(args, dataset_name, train_transform, val_transform):
    from evaluation.utils.common import classification_manifest_root

    spec = MULTILABEL_DATASETS[dataset_name]
    samples, classes, metadata = read_multilabel_manifest(
        classification_manifest_root(args) / f"{dataset_name}.json",
        args.datasets_root, dataset_name, spec["num_classes"],
    )
    return (
        MultilabelDataset(samples["train"], classes, train_transform),
        MultilabelDataset(samples["val"], classes, val_transform),
        metadata,
    )
