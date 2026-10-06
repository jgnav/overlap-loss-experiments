"""Prepare explicit available-data splits for DINOv3 video propagation.

YouTube-VOS 2019 uses its official validation set and all-frame RGB archive.
Original MOSE (2023) uses the paper's 1206/301 split sizes with a fixed seed.
These manifests do not claim to reproduce DINOv3's unpublished video lists.
The earlier MOSEv2 adaptation remains available by explicitly selecting v2.
"""

import argparse
import json
from pathlib import Path
import random
import zipfile


def video_ids(folder):
    ids = sorted(path.name for path in folder.iterdir() if path.is_dir())
    if not ids:
        raise ValueError(f"No videos found in {folder}")
    return ids


def mose_manifest(root, seed=0, release="2023", count=301):
    if release not in ("2023", "v2"):
        raise ValueError("MOSE release must be 2023 or v2")
    root = Path(root).resolve()
    folder = "MOSE2023" if release == "2023" else "MOSEv2"
    images = root / folder / "train/JPEGImages"
    masks = root / folder / "train/Annotations"
    metadata_path = root / folder / 'meta_train.json'
    catalog = None
    if release == "2023" and metadata_path.is_file():
        catalog = json.loads(metadata_path.read_text())["videos"]
    available = sorted(catalog) if catalog is not None else video_ids(images)
    if release == "2023" and (len(available) != 1507 or count != 301):
        raise ValueError("Original MOSE must contain 1507 annotated training videos for the paper's 1206/301 split")
    if not 1 <= count < len(available):
        raise ValueError("MOSE holdout must be nonempty and smaller than its annotated training release")
    if catalog is None and set(video_ids(masks)) != set(available):
        raise ValueError("MOSE RGB and annotation video IDs differ")
    shuffled = available.copy()
    random.Random(seed).shuffle(shuffled)
    held_out = sorted(shuffled[:count])
    remaining = sorted(shuffled[count:])
    if catalog is not None:
        rgb_ids, annotation_ids = set(video_ids(images)), set(video_ids(masks))
        if not set(held_out) <= rgb_ids & annotation_ids or (rgb_ids | annotation_ids) - set(available):
            raise ValueError("Selected MOSE test videos are missing or extracted video IDs differ from official metadata")
    original = release == "2023"
    return {
        "dataset": "mose", "dataset_release": release, "author_split_verified": False,
        "label": (f"MOSE 2023 seeded test ({count} videos, seed {seed})" if original
                  else f"MOSEv2 train holdout ({count} videos, seed {seed})"),
        "split_name": (f"validation1206_test301_seed{seed}" if original
                       else f"train_holdout{count}_seed{seed}"), "seed": seed,
        "source": ("Original MOSE 2023 annotated training release; seeded disjoint split matching "
                   "DINOv3's 1206 validation / 301 test sizes; author video lists unavailable" if original else
                   "Locally generated disjoint holdout from fully annotated MOSEv2 training release"),
        "dataset_source": "https://github.com/henghuiding/MOSE-api",
        "population_source": "official meta_train.json" if catalog is not None else "fully extracted training release",
        "paper_split_sizes_matched": original,
        "split_roles": {"selection": "validation", "evaluation": "test"},
        "selection_set_usage": "unused; use published DAVIS-selected propagation settings without tuning",
        "paths_relative_to": "datasets_root", "image_root": f"{folder}/train/JPEGImages",
        "mask_root": f"{folder}/train/Annotations", "frame_sampling": "all released RGB frames",
        "splits": {"selection": remaining, "evaluation": held_out},
    }


def prepare(root, output, seed=0, mose_count=301, mose_release="2023", mose_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    mose = mose_manifest(root, seed, mose_release, mose_count)
    name = f"mose2023_seed{seed}_test{mose_count}" if mose_release == "2023" else f"mosev2_seed{seed}_holdout{mose_count}"
    manifests = {name: mose}
    if not mose_only:
        manifests["youtube_vos_2019_valid"] = youtube_manifest(root)
    output.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{path}: {len(manifest['splits']['selection'])} selection, {len(manifest['splits']['evaluation'])} evaluation videos", flush=True)
    return manifests


def youtube_manifest(root):
    youtube = root / "youtube_vos_2019"
    train = video_ids(youtube / "train/JPEGImages")
    valid = video_ids(youtube / "valid/JPEGImages")
    archive = youtube / "all_frames/valid/valid_all_frames.zip"
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
    candidates = ("JPEGImages/", "valid_all_frames/JPEGImages/", "valid/JPEGImages/")
    prefix = next((candidate for candidate in candidates
                   if any(name.startswith(candidate + valid[0] + "/") for name in names)), None)
    if prefix is None:
        raise ValueError(f"Cannot locate official validation RGB videos in {archive}")
    return {
        "dataset": "youtube_vos", "dataset_release": "2019", "author_split_verified": False,
        "label": "YouTube-VOS 2019 official val (first-frame objects)", "split_name": "official_valid",
        "source": "Official YouTube-VOS 2019 validation IDs; released dense RGB archive and scoring annotations",
        "paths_relative_to": "datasets_root", "image_root": "youtube_vos_2019/valid/JPEGImages",
        "mask_root": "youtube_vos_2019/valid/Annotations",
        "rgb_archive": str(archive.relative_to(root)), "rgb_archive_prefix": prefix,
        "frame_sampling": "all released RGB frames; score annotated frames only",
        "splits": {"selection": train, "evaluation": valid},
        "selection_set_usage": "unused; use published DAVIS-selected propagation settings without tuning",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("config/video_splits"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mose-count", type=int, default=301)
    parser.add_argument("--mose-release", choices=("2023", "v2"), default="2023")
    parser.add_argument("--mose-only", action="store_true")
    args = parser.parse_args()
    prepare(args.datasets_root, args.output, args.seed, args.mose_count, args.mose_release, args.mose_only)
