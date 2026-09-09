from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _read_ids(path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _first_directory(candidates, description):
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    formatted = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {description}. Checked:\n  - {formatted}"
    )


def _find_directory(root, marker, description):
    direct = root / marker
    if direct.is_dir():
        return direct.resolve()
    matches = sorted(
        path.resolve()
        for path in root.rglob(Path(marker).name)
        if path.is_dir() and path.as_posix().endswith(marker.as_posix())
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {description} below {root}; expected */{marker}"
    )


class SegmentationDataset(Dataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if len(images) != len(targets):
            raise ValueError("Image and target lists have different lengths")
        if not images:
            raise ValueError("Segmentation dataset is empty")
        self.images = [Path(path) for path in images]
        self.targets = [Path(path) for path in targets]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def open_target(self, path):
        return Image.open(path)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = self.open_target(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def make_ade20k(root, split, transform=None, target_transform=None):
    dataset_root = _first_directory(
        [
            root / "ADEChallengeData2016",
            root / "ADE20K" / "ADEChallengeData2016",
            root / "ade20k" / "ADEChallengeData2016",
            root / "ADE20K",
            root / "ade20k",
        ],
        "ADE20K/ADEChallengeData2016",
    )
    if not (dataset_root / "images" / split).is_dir():
        dataset_root = _find_directory(
            root, Path("ADEChallengeData2016"), "ADE20K/ADEChallengeData2016"
        )
    image_dir = dataset_root / "images" / split
    target_dir = dataset_root / "annotations" / split
    images = sorted(image_dir.glob("*.jpg"))
    targets = [target_dir / f"{path.stem}.png" for path in images]
    return SegmentationDataset(images, targets, transform, target_transform)


def _resolve_voc_root(root):
    original_candidates = [
        root / "VOCdevkit" / "VOC2012",
        root / "VOC2012",
        root / "pascal_voc" / "VOCdevkit" / "VOC2012",
        root / "voc" / "VOCdevkit" / "VOC2012",
        root / "VOC" / "original" / "VOCdevkit" / "VOC2012",
        root / "VOC2012" / "original" / "VOCdevkit" / "VOC2012",
    ]
    original = None
    for candidate in original_candidates:
        if (candidate / "ImageSets" / "Segmentation" / "train.txt").is_file():
            original = candidate.resolve()
            break
    if original is None:
        matches = sorted(root.rglob("ImageSets/Segmentation/train.txt"))
        matches = [path.parents[2] for path in matches if path.parents[2].name == "VOC2012"]
        if matches:
            original = matches[0].resolve()
    if original is None:
        raise FileNotFoundError(
            f"Could not find VOCdevkit/VOC2012 below {root}"
        )

    return original


def make_pascal_voc(root, split, transform=None, target_transform=None):
    if split not in ("train", "val"):
        raise ValueError(f"Unknown PASCAL VOC split: {split}; use original train or val")
    original = _resolve_voc_root(root)
    original_train = _read_ids(
        original / "ImageSets" / "Segmentation" / "train.txt"
    )
    original_val = _read_ids(
        original / "ImageSets" / "Segmentation" / "val.txt"
    )
    validation_ids = set(original_val)
    for name, items in (("train", original_train), ("val", original_val)):
        if not items or any(not item or Path(item).name != item for item in items):
            raise ValueError(f"VOC official {name} split has empty or invalid image IDs")
        if len(set(items)) != len(items):
            raise ValueError(f"VOC official {name} split contains duplicate image IDs")
    if set(original_train) & validation_ids:
        raise ValueError("VOC official train/val splits have overlapping image IDs")
    # Preserve the official list order. SBD is neither discovered nor loaded.
    selected_ids = original_train if split == "train" else original_val
    images = [original / "JPEGImages" / f"{item}.jpg" for item in selected_ids]
    targets = [original / "SegmentationClass" / f"{item}.png" for item in selected_ids]
    dataset = SegmentationDataset(images, targets, transform, target_transform)
    ids = [path.stem for path in images]
    counts = Counter(ids)
    dataset.protocol_metadata = {
        "construction": "voc2012_original_segmentation_v1",
        "source": "VOC2012 ImageSets/Segmentation/{train,val}.txt only; no SBD",
        "mask_policy": "original VOC SegmentationClass PNG only",
        "training_order": "official split file order",
        "split": split,
        "unique_image_ids": len(counts),
        "repeated_image_entries": len(ids) - len(counts),
        "official_val_overlap_unique_ids": (
            len(set(ids).intersection(validation_ids)) if split == "train" else None
        ),
        "published_score_equivalence": (
            "Not established. User-selected original VOC splits exclude SBD, "
            "unlike the released CAPI trainaug loader."
        ),
    }
    return dataset


def segmentation_manifest(dataset):
    """Fingerprint ordered image/mask pairs, including duplicates and mask source."""
    pairs = [[str(image.resolve()), str(target.resolve())]
             for image, target in zip(dataset.images, dataset.targets, strict=True)]
    return {
        "ordered_pairs_sha256": hashlib.sha256(
            json.dumps(pairs, separators=(",", ":")).encode()
        ).hexdigest(),
        "samples": len(dataset),
        **getattr(dataset, "protocol_metadata", {}),
    }


CITYSCAPES_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
CITYSCAPES_TRAIN_ID[
    [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
] = np.arange(19, dtype=np.uint8)


class CityscapesDataset(SegmentationDataset):
    def open_target(self, path):
        raw = np.asarray(Image.open(path), dtype=np.uint8)
        return Image.fromarray(CITYSCAPES_TRAIN_ID[raw], mode="L")


def make_cityscapes(root, split, transform=None, target_transform=None):
    city_root = _first_directory(
        [
            root / "cityscapes",
            root / "Cityscapes",
            root,
        ],
        "Cityscapes",
    )
    if not (city_root / "leftImg8bit" / split).is_dir():
        markers = sorted(root.rglob("leftImg8bit"))
        if not markers:
            raise FileNotFoundError(f"Could not find Cityscapes below {root}")
        city_root = markers[0].parent.resolve()
    image_dir = city_root / "leftImg8bit" / split
    images = sorted(image_dir.glob("*/*_leftImg8bit.png"))
    targets = []
    for image in images:
        city = image.parent.name
        target_name = image.name.replace(
            "_leftImg8bit.png", "_gtFine_labelIds.png"
        )
        targets.append(city_root / "gtFine" / split / city / target_name)
    return CityscapesDataset(images, targets, transform, target_transform)


DATASET_SPECS = {
    "ade20k": {
        "display_name": "ADE20K",
        "train_split": "training",
        "test_split": "validation",
        "factory": make_ade20k,
        "ignore_labels": (0, 255),
        "num_classes": 150,
    },
    "pascal_voc": {
        "display_name": "PASCAL VOC 2012",
        "train_split": "train",
        "test_split": "val",
        "factory": make_pascal_voc,
        "ignore_labels": (255,),
        "num_classes": 21,
    },
    "cityscapes": {
        "display_name": "Cityscapes",
        "train_split": "train",
        "test_split": "val",
        "factory": make_cityscapes,
        "ignore_labels": (255,),
        "num_classes": 19,
    },
}
