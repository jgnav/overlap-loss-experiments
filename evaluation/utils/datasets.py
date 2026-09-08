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


class PascalVOCDataset(SegmentationDataset):
    def open_target(self, path):
        if path.suffix == ".mat":
            try:
                import scipy.io
            except ImportError as error:
                raise ImportError(
                    "PASCAL VOC augmented masks require scipy; install "
                    "the repository requirements.txt"
                ) from error
            array = scipy.io.loadmat(path)["GTcls"][0]["Segmentation"][0]
            return Image.fromarray(array.astype(np.uint8))
        return Image.open(path)


def _resolve_voc_roots(root):
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

    augmented_candidates = [
        root / "benchmark_RELEASE" / "dataset",
        root / "pascal_voc" / "benchmark_RELEASE" / "dataset",
        root / "voc" / "benchmark_RELEASE" / "dataset",
        root / "VOC" / "aug" / "benchmark_RELEASE" / "dataset",
        root / "VOC2012" / "aug" / "benchmark_RELEASE" / "dataset",
        original.parents[2] / "aug" / "benchmark_RELEASE" / "dataset",
    ]
    augmented = None
    for candidate in augmented_candidates:
        if (candidate / "train.txt").is_file() and (candidate / "val.txt").is_file():
            augmented = candidate.resolve()
            break
    if augmented is None:
        matches = sorted(root.rglob("benchmark_RELEASE/dataset/train.txt"))
        if matches:
            augmented = matches[0].parent.resolve()
    if augmented is None:
        raise FileNotFoundError(
            "CRISP's CAPI-style VOC protocol requires the SBD augmented masks "
            f"(benchmark_RELEASE/dataset) below {root}"
        )
    return original, augmented


def make_pascal_voc(root, split, transform=None, target_transform=None):
    original, augmented = _resolve_voc_roots(root)
    original_train = _read_ids(
        original / "ImageSets" / "Segmentation" / "train.txt"
    )
    original_val = _read_ids(
        original / "ImageSets" / "Segmentation" / "val.txt"
    )
    augmented_train = _read_ids(augmented / "train.txt")
    augmented_val = _read_ids(augmented / "val.txt")
    if split == "trainaug":
        # Literal construction in facebookresearch/capi/data.py:VOC2012.
        # CRISP A.2 specifies the released CAPI evaluator without modification.
        # Preserve list order, repeated IDs, and original/SBD mask provenance.
        # Upstream explicitly warns that this loader does not reproduce its
        # paper's VOC scores. Record duplicates and validation overlap below;
        # reproducing this code is not proof of reproducing the published scores.
        images = [original / "JPEGImages" / f"{item}.jpg" for item in original_train]
        targets = [original / "SegmentationClass" / f"{item}.png" for item in original_train]
        images.extend(augmented / "img" / f"{item}.jpg" for item in augmented_train + augmented_val)
        targets.extend(augmented / "cls" / f"{item}.mat" for item in augmented_train + augmented_val)
    elif split == "val":
        images = [original / "JPEGImages" / f"{item}.jpg" for item in original_val]
        targets = [
            original / "SegmentationClass" / f"{item}.png"
            for item in original_val
        ]
    else:
        raise ValueError(f"Unknown PASCAL VOC split: {split}")
    dataset = PascalVOCDataset(images, targets, transform, target_transform)
    ids = [path.stem for path in images]
    counts = Counter(ids)
    dataset.protocol_metadata = {
        "construction": "capi_released_voc2012_trainaug_v1",
        "source": "https://github.com/facebookresearch/capi/blob/main/data.py",
        "split": split,
        "unique_image_ids": len(counts),
        "repeated_image_entries": len(ids) - len(counts),
        "official_val_overlap_unique_ids": (
            len(set(ids).intersection(original_val)) if split == "trainaug" else None
        ),
        "upstream_limitation": (
            "CAPI's released VOC loader states it does not reproduce its paper's "
            "results. Its trainaug concatenation retains repeated image IDs and "
            "any official validation images present in SBD."
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
        "train_split": "trainaug",
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
