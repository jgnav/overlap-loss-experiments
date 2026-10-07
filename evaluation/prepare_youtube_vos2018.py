"""Download and prepare the official YouTube-VOS 2018 release on a CPU node.

Archives and downloads live outside the project quota. Publication is atomic;
partial downloads and extraction stages are reusable after interruption.
This does not claim to recover DINOv3's unpublished split or filtering rule.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import time
import zipfile

import requests


SOURCE = "https://drive.google.com/drive/folders/1L9JAl1BCtzomRJ34tKsU9tJUmF-ZMFOM"
TRAIN_IDS = [
    "1dBwodbs1d6tLgBAe-dMarGv81Gs2vwPQ", "1BAcGzbVj-A9X4NiBJPJji2d6RZXpxgob",
    "1Gs1XpSvQx3LYKwIg1Oa6watdNz3hVNep", "1zkHbPURefpuvB55nq4ANzHKxuJOtT4oL",
    "1PRIOAQHhzav76kZSgw24dqkCehdukYYI", "186-Ag5wSeUkoLDBsr1oTkim-6JS-SMxz",
    "1D5f3dtkhof0bsLlJjMOHl5yzV_PkBidA", "1v4pthddPAMwPYE6ABFJPRjsi6-DpKnz1",
    "1Hr7kdcWns3QufhFWT-nBnP3FEUjh1GVv", "1mIuqf7Roy4sVM7P5YgvQUSk8yPg8isPm",
    "1nGMNkYoTfM2MXh3fYfJjxgMnoYBx8gHl", "1NVrJG9aTs94bC1Sfs3Rwcaffrz69hII6",
]
DENSE_VALID_IDS = [
    "1HKbUoRwy9tNaMs0P4earfYyO8iu12363", "1mpUxL0P_cMNcEUiTXQXFwEii7jHaZH_d",
    "1voYwdHQJhWWXEmLjRiEYdnwk9tBNYEe-", "1rwdElYM_QE-iZcWlzod_UcjsYgqD8vW9",
    "1hZZas9XwVPZCwMfLFgLuOqslxJGqCz8m", "1DgsfAy5RAibzJC2qW7PYHOgL8a-L79kH",
    "1zQGTzGJBOMWFiBiUpoWLrIGW_meDHrqZ", "1j7zVmzmKBs3KtxZJGnrrJO4nplEz2ptb",
]
FILES = ([dict(name=f"train.7z.{i:03d}", id=fid) for i, fid in enumerate(TRAIN_IDS, 1)]
         + [dict(name="valid.zip", id="1HWYO5Ii476Z6fkhKv05WWRzoHVDBKp-t"),
            dict(name="validation_gt.zip", id="1gXelzNfABdFN9oD2_6bXuu8bKxzUD5JH")]
         + [dict(name=f"valid_all_frames.7z.{i:03d}", id=fid)
            for i, fid in enumerate(DENSE_VALID_IDS, 1)])

# Public mirrors of the unmodified validation archives. Pin their published
# SHA256 values and still use the official released scoring annotations.
VALIDATION_FILES = [
    dict(name="valid.zip", id="1HWYO5Ii476Z6fkhKv05WWRzoHVDBKp-t",
         url="https://huggingface.co/datasets/xwsa568/ytvos18/resolve/main/valid.zip?download=true",
         expected_sha256="b2af7fbc88e403fd3cff0cbf11016d9953481f9b6abf5b978c0d36f316403a53"),
    dict(name="validation_gt.zip", id="1gXelzNfABdFN9oD2_6bXuu8bKxzUD5JH"),
    dict(name="valid_all_frames.zip", id="hf:xwsa568/ytvos18:valid_all_frames.zip",
         url="https://huggingface.co/datasets/xwsa568/ytvos18/resolve/main/valid_all_frames.zip?download=true",
         expected_sha256="7906bd0d186c2f0202bd10447ba7b39b56fd8889a1ea0d8164ebf887328fc9f9"),
]


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, directory):
    target = directory / entry["name"]
    marker = target.with_name(target.name + ".download.json")
    if marker.is_file() and target.is_file():
        previous = json.loads(marker.read_text())
        if previous["id"] == entry["id"] and previous["bytes"] == target.stat().st_size:
            log(f"Reuse downloaded {target.name}")
            return previous
    partial = target.with_name(target.name + ".partial")
    url = entry.get("url", f"https://drive.usercontent.google.com/download?id={entry['id']}&export=download&confirm=t")
    # Drive may reject unbounded requests for popular files while serving
    # ordinary bounded ranges. Verify each range before appending its bytes.
    size_marker = partial.with_name(partial.name + ".size.json")
    expected = json.loads(size_marker.read_text())["bytes"] if size_marker.exists() else None
    failures = 0
    while True:
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if expected is not None and offset > expected:
                raise RuntimeError(f"Partial download exceeds archive size: {offset} > {expected}")
            if expected is not None and offset == expected:
                partial.replace(target)
                record = {**entry, "bytes": offset, "sha256": sha256(target), "source_folder": SOURCE}
                if entry.get("expected_sha256") and record["sha256"] != entry["expected_sha256"]:
                    raise RuntimeError(f"Archive SHA256 mismatch: {target.name}")
                write_json(marker, record)
                log(f"Downloaded {target.name}: {offset} bytes")
                return record
            end = offset + (32 << 20) - 1
            if expected is not None:
                end = min(end, expected - 1)
            headers = {"Range": f"bytes={offset}-{end}"}
            log(f"Download {target.name}, bytes {offset}-{end}")
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("Drive returned an HTML permission/quota page instead of archive bytes")
                if response.status_code != 206:
                    raise RuntimeError(f"Server did not honor bounded range: HTTP {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                unit, limits = content_range.split(" ", 1)
                interval, total = limits.split("/")
                start, actual_end = map(int, interval.split("-"))
                total = int(total)
                if unit != "bytes" or start != offset or actual_end != min(end, total - 1):
                    raise RuntimeError(f"Unexpected resume range: {content_range}")
                if expected is not None and expected != total:
                    raise RuntimeError("Remote archive size changed during download")
                expected = total
                write_json(size_marker, {"id": entry["id"], "bytes": expected})
                with partial.open("ab") as output:
                    for block in response.iter_content(4 << 20):
                        output.write(block)
                        offset += len(block)
                if offset != actual_end + 1:
                    raise RuntimeError(f"Truncated range: {offset} of {actual_end + 1} bytes")
            failures = 0
        except (requests.RequestException, RuntimeError, OSError, ValueError) as error:
            failures += 1
            log(f"Download attempt {failures} failed for {target.name}: {error}")
            if failures >= 8:
                raise
            time.sleep(min(120, 15 * 2 ** (failures - 1)))


def joined_archive(directory, name, count):
    parts = [directory / f"{name}.{i:03d}" for i in range(1, count + 1)]
    with parts[0].open("rb") as source:
        header = source.read(32)
    if header[:6] != b"7z\xbc\xaf\x27\x1c":
        raise ValueError(f"Not a 7z archive: {parts[0]}")
    _, offset, size, _ = struct.unpack("<IQQI", header[8:])
    expected = 32 + offset + size
    if sum(path.stat().st_size for path in parts) != expected:
        raise ValueError(f"Missing or truncated {name} archive volumes")
    target = directory / name
    if not target.is_file() or target.stat().st_size != expected:
        partial = target.with_name(target.name + ".partial")
        with partial.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, 8 << 20)
        partial.replace(target)
    return target


def extract(archive, destination, bsdtar):
    marker = destination / ".extraction_complete"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    log(f"Extract and verify CRC: {archive.name}")
    # libarchive verifies archive checksums while extracting. Official archives
    # are kept, and failed extractions remain isolated from the installed data.
    subprocess.run([bsdtar, "-xf", str(archive), "-C", str(destination), "--no-same-owner"], check=True)
    marker.write_text(str(archive) + "\n")


def expand(archive, destination, bsdtar):
    extract(archive, destination, bsdtar)
    # The official split 7z files may contain a ZIP rather than video folders.
    for depth in range(3):
        pending = []
        for folder, dirs, files in os.walk(destination):
            dirs[:] = [name for name in dirs if name not in {"JPEGImages", "Annotations"}]
            for name in files:
                path = Path(folder) / name
                if path.suffix.lower() in {".zip", ".7z", ".tar", ".gz"}:
                    nested = path.with_name(path.name + ".unpacked")
                    if not (nested / ".extraction_complete").exists():
                        pending.append((path, nested))
        if not pending:
            return
        for path, nested in pending:
            extract(path, nested, bsdtar)
    raise ValueError("Unexpected excessive nested archives")


def find_directory(root, name):
    found = []
    for folder, dirs, _ in os.walk(root):
        if name in dirs:
            found.append(Path(folder) / name)
        dirs[:] = [item for item in dirs if item not in {"JPEGImages", "Annotations", "__MACOSX"}]
    if len(found) != 1:
        raise ValueError(f"Expected one {name} directory in {root}, found {found}")
    return found[0]


def install_directory(source, target):
    if target.is_dir():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


def inventory(split, expected_count, full_gt):
    images, masks = split / "JPEGImages", split / "Annotations"
    metadata = json.loads((split / "meta.json").read_text())
    videos = metadata["videos"]
    rgb_ids = {path.name for path in images.iterdir() if path.is_dir()}
    mask_ids = {path.name for path in masks.iterdir() if path.is_dir()}
    if len(videos) != expected_count or set(videos) != rgb_ids or rgb_ids != mask_ids:
        raise ValueError(f"Unexpected video population for {split}: metadata={len(videos)}, RGB={len(rgb_ids)}, masks={len(mask_ids)}")
    counts = {"videos": len(videos), "rgb_frames": 0, "annotation_frames": 0}
    for name, video in videos.items():
        frames = {p.stem for p in (images / name).glob("*.jpg")}
        labels = {p.stem for p in (masks / name).glob("*.png")}
        required = {frame for obj in video["objects"].values() for frame in obj["frames"]}
        if len(frames) < 2 or not labels or required - frames or labels - frames:
            raise ValueError(f"Missing RGB or initial masks: {name}")
        if full_gt and required - labels:
            raise ValueError(f"Incomplete scoring annotations: {name}")
        counts["rgb_frames"] += len(frames)
        counts["annotation_frames"] += len(labels)
    return counts, sorted(videos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bsdtar", default=shutil.which("bsdtar"))
    parser.add_argument("--phase", choices=("all", "validation"), default="all")
    parser.add_argument("--download-retry-hours", type=float, default=0)
    args = parser.parse_args()
    root = args.datasets_root.resolve()
    work = root / "downloads/youtube_vos2018"
    work.mkdir(parents=True, exist_ok=True)
    destination = root / "youtube_vos_2018"
    if (destination / "preparation_report.json").is_file():
        log(f"Dataset already prepared: {destination}")
        return
    if args.phase == "validation" and (destination / "validation_preparation_report.json").is_file():
        log(f"Validation already prepared: {destination}")
        return
    if not args.bsdtar or not 1 <= args.workers <= 4:
        raise ValueError("bsdtar and 1–4 download workers are required")
    files = VALIDATION_FILES if args.phase == "validation" else FILES[:12] + VALIDATION_FILES
    write_json(work / "source_catalog.json", {"release": "2018", "folder": SOURCE, "files": files})
    deadline = time.monotonic() + args.download_retry_hours * 3600
    while True:
        records, errors = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, entry, work): entry for entry in files}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as error:
                    errors.append({"file": futures[future]["name"], "error": str(error)})
                write_json(work / "download_progress.json", {
                    "phase": args.phase, "completed": len(records), "total": len(files),
                    "files": records, "errors": errors,
                })
        if not errors:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Downloads still unavailable: {errors}")
        log(f"{len(errors)} archives unavailable; retry in 20 minutes, retaining completed files")
        time.sleep(min(1200, max(0, deadline - time.monotonic())))
    stage = root / "youtube_vos_2018.installing"
    stage.mkdir(exist_ok=True)
    if destination.exists():
        if not (destination / "validation_preparation_report.json").is_file():
            raise FileExistsError(f"Refusing to replace existing dataset: {destination}")
        for name in ("valid", "all_frames"):
            target = stage / name
            if not target.exists():
                target.symlink_to(destination / name, target_is_directory=True)
        (stage / ".full_valid_gt").touch()
    archives = {"valid": work / "valid.zip",
                "validation_gt": work / "validation_gt.zip",
                "valid_all_frames": work / "valid_all_frames.zip"}
    if args.phase == "all":
        archives["train"] = joined_archive(work, "train.7z", 12)
    for name, archive in archives.items():
        extracted = work / f"extracted_{name}"
        expand(archive, extracted, args.bsdtar)
        if name in {"train", "valid"}:
            rgb_target = stage / name / "JPEGImages"
            if not rgb_target.exists():
                rgb = find_directory(extracted, "JPEGImages")
                metadata = rgb.parent / "meta.json"
                install_directory(rgb, rgb_target)
                shutil.copy2(metadata, stage / name / "meta.json")
            if not (stage / name / "Annotations").exists():
                install_directory(find_directory(extracted, "Annotations"), stage / name / "Annotations")
        elif name == "validation_gt":
            # Full released GT replaces initialization-only valid masks.
            if not (stage / ".full_valid_gt").exists():
                gt = find_directory(extracted, "Annotations")
                shutil.copytree(gt, stage / "valid/Annotations", dirs_exist_ok=True)
                (stage / ".full_valid_gt").write_text("validation_gt.zip; not sample predictions\n")
        else:
            dense = stage / "all_frames/valid/JPEGImages"
            if not dense.exists():
                install_directory(find_directory(extracted, "JPEGImages"), dense)
    valid, valid_ids = inventory(stage / "valid", 474, True)
    dense = stage / "all_frames/valid/JPEGImages"
    if {p.name for p in dense.iterdir() if p.is_dir()} != set(valid_ids):
        raise ValueError("Dense validation video IDs differ from official 2018 validation")
    dense_count = 0
    for name in valid_ids:
        frames = {p.stem for p in (dense / name).glob("*.jpg")}
        masks = {p.stem for p in (stage / "valid/Annotations" / name).glob("*.png")}
        if masks - frames:
            raise ValueError(f"Dense validation RGB is missing scoring frames: {name}")
        dense_count += len(frames)
    if args.phase == "validation":
        write_json(stage / "official_valid.json", {
            "dataset": "youtube_vos", "dataset_release": "2018",
            "author_split_verified": False, "paper_split_sizes_matched": False,
            "split_name": "official_valid", "label": "YouTube-VOS 2018 official validation (474 videos)",
            "source": "Official 2018 validation IDs with released full validation_gt.zip and all-frame RGB",
            "paths_relative_to": "datasets_root", "image_root": "youtube_vos_2018/all_frames/valid/JPEGImages",
            "mask_root": "youtube_vos_2018/valid/Annotations",
            "frame_sampling": "all released RGB frames; score annotated frames only",
            "splits": {"selection": [], "evaluation": valid_ids},
        })
        write_json(stage / "validation_preparation_report.json", {
            "status": "validation_ready", "release": "2018", "valid": valid,
            "dense_valid_rgb_frames": dense_count, "archives": records,
            "training_holdout_ready": False, "full_preparation_complete": False,
        })
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing dataset: {destination}")
        stage.rename(destination)
        log(f"Verified full validation data ready; training archives still pending: {destination}")
        return
    train, train_ids = inventory(stage / "train", 3471, True)
    # Do not discard 23 arbitrary videos to imply recovery of the paper's
    # 3448-video population. Preserve the full release and label the difference.
    shuffled = train_ids.copy()
    random.Random(0).shuffle(shuffled)
    manifest = {
        "dataset": "youtube_vos", "dataset_release": "2018", "author_split_verified": False,
        "paper_split_sizes_matched": False, "seed": 0, "split_name": "train_holdout690_seed0",
        "label": "YouTube-VOS 2018 seeded train holdout (690 videos)",
        "source": "Official 3471-video training release; seeded 2781/690 split. DINOv3's 3448-video population filtering and IDs are unpublished.",
        "dataset_source": SOURCE, "paths_relative_to": "datasets_root",
        "image_root": "youtube_vos_2018/train/JPEGImages", "mask_root": "youtube_vos_2018/train/Annotations",
        "frame_sampling": "official training RGB release; dense train archive not supplied in current official folder",
        "selection_set_usage": "unused; no hyperparameter tuning performed",
        "splits": {"selection": sorted(shuffled[690:]), "evaluation": sorted(shuffled[:690])},
    }
    write_json(stage / "train_holdout690_seed0.json", manifest)
    write_json(stage / "official_valid.json", {
        **manifest, "split_name": "official_valid", "label": "YouTube-VOS 2018 official validation (474 videos)",
        "source": "Official 2018 validation IDs with released validation_gt.zip scoring masks and all-frame RGB",
        "image_root": "youtube_vos_2018/all_frames/valid/JPEGImages", "mask_root": "youtube_vos_2018/valid/Annotations",
        "frame_sampling": "all released RGB frames; score annotated frames only",
        "splits": {"selection": train_ids, "evaluation": valid_ids},
    })
    report = {"status": "complete", "release": "2018", "source": SOURCE, "path": str(destination),
              "train": train, "valid": valid, "dense_valid_rgb_frames": dense_count,
              "archives": records, "archive_checks": "7z length/header and extraction CRC; download SHA256 recorded",
              "dino_v3_author_split_verified": False, "dino_v3_population_difference": "3471 released vs 3448 in paper",
              "dense_train_frames_available": False, "offline_scoring_ready": True}
    write_json(stage / "preparation_report.json", report)
    if destination.exists():
        # Only extend the verified validation-only installation; publish the
        # complete train directory and completion marker after all checks.
        if (destination / "train").exists():
            raise FileExistsError(f"Refusing to replace existing train data: {destination}")
        (stage / "train").rename(destination / "train")
        for name in ("train_holdout690_seed0.json", "official_valid.json", "preparation_report.json"):
            (stage / name).replace(destination / name)
    else:
        stage.rename(destination)
    log(f"Prepared and verified YouTube-VOS 2018: {destination}")


if __name__ == "__main__":
    main()
