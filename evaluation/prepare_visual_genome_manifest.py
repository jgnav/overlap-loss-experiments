"""Convert public SSGRL VG500 lists/indices to the offline probe manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.utils.classification_data import read_multilabel_manifest
from evaluation.utils.common import write_json


NUM_CLASSES = 500
FILES = ("train_list_500.txt", "test_list_500.txt", "vg_category_500_labels_index.json")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_image(datasets_root, filename):
    if Path(filename).name != filename or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise ValueError(f"Unsafe or unsupported VG500 image name: {filename}")
    candidates = [datasets_root / "visual_genome" / subdir / filename for subdir in
                  ("VG_100K", "VG_100K_2", "images", "")]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"Expected one VG500 image for {filename}; found {existing}")
    return existing[0].relative_to(datasets_root).as_posix()


def build_manifest(datasets_root, annotations_dir):
    datasets_root = Path(datasets_root).expanduser().resolve()
    annotations_dir = Path(annotations_dir).expanduser().resolve()
    paths = {name: annotations_dir / name for name in FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SSGRL VG500 annotation files: {missing}")
    labels_by_image = json.loads(paths[FILES[2]].read_text(encoding="utf-8"))
    if not isinstance(labels_by_image, dict):
        raise ValueError("VG500 labels index must be a filename-to-index-list mapping")
    splits, seen = {}, set()
    for target, filename in (("train", FILES[0]), ("val", FILES[1])):
        names = [line.strip() for line in paths[filename].read_text().splitlines() if line.strip()]
        if not names:
            raise ValueError(f"Empty VG500 {target} split")
        rows = []
        for name in names:
            if name in seen:
                raise ValueError(f"Duplicate VG500 image or train/test overlap: {name}")
            seen.add(name)
            indices = labels_by_image.get(name)
            if (not isinstance(indices, list) or not indices
                    or any(type(index) is not int or not 0 <= index < NUM_CLASSES for index in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError(f"Invalid 500-class positive indices for {name}")
            rows.append({"id": name, "image": _find_image(datasets_root, name),
                         "positive_indices": indices})
        splits[target] = rows
    return {
        "dataset": "visual_genome",
        "source": "SSGRL public VisualGenome-500 train_list_500/test_list_500; test used as evaluation split",
        "source_files_sha256": {name: _sha256(path) for name, path in paths.items()},
        "classes": [f"VG500_{index:03d}" for index in range(NUM_CLASSES)],
        "splits": splits,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True,
                        help="Directory containing SSGRL's three data/VG annotation files")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.datasets_root.expanduser().resolve()
    output = (args.output or root / "evaluation_manifests/visual_genome.json").expanduser().resolve()
    manifest = build_manifest(root, args.annotations_dir)
    if output.is_file():
        if json.loads(output.read_text()) != manifest:
            raise FileExistsError(f"Existing VG500 manifest differs: {output}")
    else:
        write_json(output, manifest)
    read_multilabel_manifest(output, root, "visual_genome", NUM_CLASSES)
    print(f"VG500 manifest ready: {output}; {len(manifest['splits']['train'])} train, "
          f"{len(manifest['splits']['val'])} evaluation images")


if __name__ == "__main__":
    main()
