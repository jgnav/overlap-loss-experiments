"""Prepare image-level, 80-class labels from official COCO 2017 instances."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def build_coco2017_manifest(datasets_root, coco_root=None):
    datasets_root = Path(datasets_root).expanduser().resolve()
    coco_root = (
        Path(coco_root).expanduser().resolve() if coco_root is not None
        else datasets_root / "coco"
    )
    splits, hashes, statistics = {}, {}, {}
    vocabulary, seen_images = None, set()
    for split in ("train", "val"):
        annotation_path = coco_root / "annotations" / f"instances_{split}2017.json"
        raw = annotation_path.read_bytes()
        hashes[annotation_path.name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        del raw
        categories = data["categories"]
        current_vocabulary = {category["id"]: category["name"] for category in categories}
        if (len(categories) != 80 or len(current_vocabulary) != 80
                or len(set(current_vocabulary.values())) != 80
                or any(type(key) is not int or key <= 0 for key in current_vocabulary)
                or any(not isinstance(name, str) or not name.strip() for name in current_vocabulary.values())):
            raise ValueError(f"{annotation_path}: expected 80 unique COCO categories")
        if vocabulary is not None and current_vocabulary != vocabulary:
            raise ValueError("COCO train/val category vocabularies differ")
        vocabulary = current_vocabulary
        category_ids = sorted(vocabulary)
        category_index = {key: index for index, key in enumerate(category_ids)}
        samples = {}
        for image in data["images"]:
            image_id = image["id"]
            if type(image_id) is not int or image_id < 0:
                raise ValueError(f"{annotation_path}: invalid image ID")
            if image_id in seen_images:
                raise ValueError(f"COCO duplicate image ID or train/val overlap: {image_id}")
            seen_images.add(image_id)
            filename = image["file_name"]
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".jpg"):
                raise ValueError(f"{annotation_path}: expected a JPEG filename, not a path")
            image_path = coco_root / "images" / f"{split}2017" / filename
            try:
                image_path = image_path.relative_to(datasets_root)
            except ValueError:
                pass
            samples[image_id] = {
                "id": str(image_id), "image": image_path.as_posix(), "labels": [0] * 80,
            }
        crowd_instances = 0
        for annotation in data["annotations"]:
            image_id, category_id = annotation["image_id"], annotation["category_id"]
            if image_id not in samples or category_id not in category_index:
                raise ValueError(f"{annotation_path}: annotation references an unknown image/category")
            # A crowd still establishes category presence at image level.
            samples[image_id]["labels"][category_index[category_id]] = 1
            crowd_instances += int(annotation.get("iscrowd", 0) == 1)
        splits[split] = [samples[key] for key in sorted(samples)]
        statistics[split] = {
            "images": len(samples),
            "instances": len(data["annotations"]),
            "crowd_instances_included": crowd_instances,
            "images_without_annotations_retained": sum(not any(row["labels"]) for row in samples.values()),
        }
        del data, samples
    return {
        "dataset": "coco",
        "source": (
            "User-selected MS-COCO 2017 official train2017/val2017 image splits and "
            "instances_train2017.json/instances_val2017.json. Image-level category "
            "presence: 1 if any annotated instance is present (including iscrowd=1), "
            "0 otherwise. All images retained, including all-negative images. "
            "Classes ordered by ascending original COCO category ID, not ID minus one. "
            "Exact CG-SSL/CRISP split equivalence is not established."
        ),
        "classes": [vocabulary[key] for key in category_ids],
        "category_ids": category_ids,
        "source_files_sha256": hashes,
        "split_statistics": statistics,
        "splits": splits,
    }


def prepare_coco2017_manifest(datasets_root, coco_root=None, output=None):
    from evaluation.utils.classification_data import read_multilabel_manifest

    datasets_root = Path(datasets_root).expanduser().resolve()
    output = (
        Path(output).expanduser().resolve() if output is not None
        else datasets_root / "evaluation_manifests/coco.json"
    )
    manifest = build_coco2017_manifest(datasets_root, coco_root)
    # Compact encoding keeps the 123,287 image-level label vectors manageable.
    payload = (json.dumps(manifest, separators=(",", ":")) + "\n").encode("utf-8")
    del manifest
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError(f"Refusing to overwrite a different manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem so publication is atomic: preflight
    # must never see a partially written JSON file, even during a concurrent job.
    with tempfile.TemporaryDirectory(prefix=".coco-manifest-", dir=output.parent) as directory:
        draft = Path(directory) / "coco.json"
        draft.write_bytes(payload)
        _, _, metadata = read_multilabel_manifest(draft, datasets_root, "coco", 80)
        # BeeGFS/sandbox mounts can reject hard links. An exclusive directory
        # serializes publishers; rename then exposes the complete file at once.
        lock = output.with_name(f".{output.name}.publish-lock")
        try:
            lock.mkdir()
        except FileExistsError:
            raise FileExistsError(f"Manifest publication lock exists: {lock}; another preparer may be running") from None
        try:
            if output.exists():
                if output.read_bytes() != payload:
                    raise FileExistsError(f"Refusing to overwrite a different manifest: {output}")
            else:
                draft.replace(output)
        finally:
            lock.rmdir()
    metadata["manifest"] = str(output)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("dataset"))
    parser.add_argument("--coco-root", type=Path, help="COCO root containing images/ and annotations/")
    parser.add_argument("--output", type=Path, help="Default: DATASETS_ROOT/evaluation_manifests/coco.json")
    args = parser.parse_args(argv)
    print("Preparing COCO 2017 labels and validating every image path (CPU only)...", flush=True)
    metadata = prepare_coco2017_manifest(args.datasets_root, args.coco_root, args.output)
    print(f"Validated COCO2017 classification manifest: {metadata['manifest']}", flush=True)
    print(f"Split sizes: {metadata['split_sizes']}; classes: {len(metadata['classes'])}", flush=True)
    print(f"Manifest SHA256: {metadata['sha256']}", flush=True)


if __name__ == "__main__":
    main()
