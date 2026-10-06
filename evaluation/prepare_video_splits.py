"""Prepare explicit available-data splits for DINOv3 video propagation.

YouTube-VOS 2019 uses its official validation set and all-frame RGB archive.
MOSEv2 uses a deterministic held-out subset of its fully annotated training
release. These manifests do not claim to reproduce DINOv3's unpublished splits.
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


def prepare(root, output, seed=0, mose_count=301):
    root, output = Path(root).resolve(), Path(output).resolve()
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
    available = video_ids(root / "MOSEv2/train/JPEGImages")
    if not 1 <= mose_count < len(available):
        raise ValueError("MOSE holdout must be nonempty and smaller than its annotated training release")
    shuffled = available.copy()
    random.Random(seed).shuffle(shuffled)
    held_out = sorted(shuffled[:mose_count])
    remaining = sorted(shuffled[mose_count:])
    manifests = {
        "youtube_vos_2019_valid": {
            "dataset": "youtube_vos", "dataset_release": "2019", "author_split_verified": False,
            "label": "YouTube-VOS 2019 official val (first-frame objects)", "split_name": "official_valid",
            "source": "Official YouTube-VOS 2019 validation IDs; released dense RGB archive and scoring annotations",
            "paths_relative_to": "datasets_root", "image_root": "youtube_vos_2019/valid/JPEGImages",
            "mask_root": "youtube_vos_2019/valid/Annotations",
            "rgb_archive": str(archive.relative_to(root)), "rgb_archive_prefix": prefix,
            "frame_sampling": "all released RGB frames; score annotated frames only",
            "splits": {"selection": train, "evaluation": valid},
        },
        f"mosev2_seed{seed}_holdout{mose_count}": {
            "dataset": "mose", "dataset_release": "v2", "author_split_verified": False,
            "label": f"MOSEv2 train holdout ({mose_count} videos, seed {seed})",
            "split_name": f"train_holdout{mose_count}_seed{seed}", "seed": seed,
            "source": "Locally generated disjoint holdout from fully annotated MOSEv2 training release; published propagation parameters, no tuning",
            "paths_relative_to": "datasets_root", "image_root": "MOSEv2/train/JPEGImages",
            "mask_root": "MOSEv2/train/Annotations", "frame_sampling": "all released RGB frames",
            "splits": {"selection": remaining, "evaluation": held_out},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        manifest["selection_set_usage"] = "unused; use published DAVIS-selected propagation settings without tuning"
        path = output / f"{name}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{path}: {len(manifest['splits']['evaluation'])} evaluation videos", flush=True)
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("config/video_splits"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mose-count", type=int, default=301)
    args = parser.parse_args()
    prepare(args.datasets_root, args.output, args.seed, args.mose_count)
