"""Prepare explicit VOC2012 classification train/val inputs (no downloads)."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
VOC_LABELS = {"-1": 0, "0": None, "1": 1}


def build_voc2012_manifest(datasets_root, voc_root=None):
    datasets_root = Path(datasets_root).expanduser().resolve()
    voc_root = (
        Path(voc_root).expanduser().resolve() if voc_root is not None
        else datasets_root / "pascal_voc/VOCdevkit/VOC2012"
    )
    main = voc_root / "ImageSets/Main"
    source_hashes = {}

    def read_source(path):
        raw = path.read_bytes()
        source_hashes[path.relative_to(voc_root).as_posix()] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8").splitlines()

    splits, seen_ids = {}, set()
    for split in ("train", "val"):
        split_path = main / f"{split}.txt"
        ids = [line.strip() for line in read_source(split_path) if line.strip()]
        if not ids or any(Path(item).name != item or len(item.split()) != 1 for item in ids):
            raise ValueError(f"{split_path}: expected a nonempty list of image IDs")
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError(f"{split_path}: duplicate image IDs or train/val overlap")
        seen_ids.update(ids)
        labels = {item: [] for item in ids}
        for class_name in VOC_CLASSES:
            annotation = main / f"{class_name}_{split}.txt"
            class_labels = {}
            for line in read_source(annotation):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2 or fields[1] not in VOC_LABELS:
                    raise ValueError(f"{annotation}: expected image ID and native label -1/0/1")
                item, native_label = fields
                if item in class_labels:
                    raise ValueError(f"{annotation}: duplicate image ID {item}")
                class_labels[item] = VOC_LABELS[native_label]
            if set(class_labels) != set(ids):
                raise ValueError(f"{annotation}: image IDs do not match {split_path.name}")
            for item in ids:
                labels[item].append(class_labels[item])
        splits[split] = []
        for item in ids:
            image = voc_root / "JPEGImages" / f"{item}.jpg"
            try:
                image = image.relative_to(datasets_root)
            except ValueError:
                pass  # An explicitly supplied external VOC root needs absolute paths.
            splits[split].append({"id": item, "image": image.as_posix(), "labels": labels[item]})
    return {
        "dataset": "pascal_voc",
        "source": (
            "PASCAL VOC2012 official classification ImageSets/Main train and val; "
            "20 per-class annotation files per split. Native -1=negative maps to 0; "
            "0=difficult maps to null; 1=positive maps to 1. No SBD/segmentation lists."
        ),
        "classes": list(VOC_CLASSES),
        "source_files_sha256": source_hashes,
        "splits": splits,
    }


def prepare_voc2012_manifest(datasets_root, voc_root=None, output=None):
    from evaluation.utils.classification_data import read_multilabel_manifest

    datasets_root = Path(datasets_root).expanduser().resolve()
    output = (
        Path(output).expanduser().resolve() if output is not None
        else datasets_root / "evaluation_manifests/pascal_voc.json"
    )
    manifest = build_voc2012_manifest(datasets_root, voc_root)
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError(f"Refusing to overwrite a different manifest: {output}")

    # Run the evaluator's complete validation before publishing any new input.
    with tempfile.TemporaryDirectory(prefix="voc-manifest-") as directory:
        draft = Path(directory) / "pascal_voc.json"
        draft.write_bytes(payload)
        _, _, metadata = read_multilabel_manifest(draft, datasets_root, "pascal_voc", len(VOC_CLASSES))
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            if output.read_bytes() != payload:
                raise FileExistsError(f"Refusing to overwrite a different manifest: {output}") from None
    metadata["manifest"] = str(output)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("dataset"))
    parser.add_argument("--voc-root", type=Path, help="Extracted VOC2012 directory containing ImageSets/Main")
    parser.add_argument("--output", type=Path, help="Default: DATASETS_ROOT/evaluation_manifests/pascal_voc.json")
    args = parser.parse_args(argv)
    metadata = prepare_voc2012_manifest(args.datasets_root, args.voc_root, args.output)
    print(f"Validated VOC2012 classification manifest: {metadata['manifest']}", flush=True)
    print(f"Split sizes: {metadata['split_sizes']}; classes: {len(metadata['classes'])}", flush=True)
    print(f"Manifest SHA256: {metadata['sha256']}", flush=True)


if __name__ == "__main__":
    main()
