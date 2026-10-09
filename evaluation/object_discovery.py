"""DINOv3's published TokenCut evaluation for frozen ViT-S/16 backbones.

Prepare validates every evaluation image and annotation and freezes source code.
Run assigns one model/dataset to each Slurm array task and resumes per image.
Collect reports every threshold and the best dataset-level CorLoc in the sweep.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sqlite3
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED = {"VOC07": 5011, "VOC12": 11540, "COCO20K": 19817}
SOURCES = {
    "paper": "https://arxiv.org/abs/2508.10104",
    "protocol": "https://arxiv.org/html/2508.10104v1#A4.SS4",
    "dinov3_public_commit": "6876159a11b4df116f30f667f8c9888617df0751",
    "tokencut": "https://github.com/YangtaoWANG95/TokenCut/tree/fed52cd5b60891baefd8ec7110dafa73be816ee1",
    "dino_weights": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    from evaluation.utils.object_discovery import THRESHOLDS
    expected = {
        "patch_size": 16, "features": "final_norm_patch_output",
        "preprocessing": "native_resolution_zero_pad", "thresholds": list(THRESHOLDS),
        "eps": 1e-5, "iou_threshold": 0.5, "iou_comparison": "greater_than",
        "voc_include_difficult_and_truncated": True, "coco_remove_crowd": True,
    }
    if config["protocol"] != expected:
        raise ValueError("Configuration must preserve the extended-sweep protocol: " + str(expected))
    if config["datasets"] != list(EXPECTED):
        raise ValueError("Evaluate all three official datasets in VOC07/VOC12/COCO20K order")


def dataset_records(config):
    from PIL import Image
    from evaluation.utils.object_discovery import coco_boxes, voc_boxes
    data = Path(config["datasets_root"])
    datasets, inputs = {}, {}
    for name, year in (("VOC07", "2007"), ("VOC12", "2012")):
        print("Reading", name, "trainval annotations", flush=True)
        root = data / "pascal_voc/VOCdevkit" / ("VOC" + year)
        layout = root / "ImageSets/Main/trainval.txt"
        ids = layout.read_text().split()
        if len(ids) != EXPECTED[name] or len(set(ids)) != len(ids):
            raise ValueError(f"Incorrect {name} trainval split: {len(ids)}")
        rows, annotations_hash = [], hashlib.sha256()
        def read_voc(image_id):
            annotation_path = root / "Annotations" / (image_id + ".xml")
            annotation_bytes = annotation_path.read_bytes()
            annotation = ET.fromstring(annotation_bytes)
            width, height = [int(annotation.findtext("size/" + k)) for k in ("width", "height")]
            boxes = voc_boxes(annotation)
            if not boxes:
                raise ValueError("Missing VOC ground truth: " + str(annotation_path))
            return ({"id": image_id, "image": str(root / "JPEGImages" / (image_id + ".jpg")),
                     "width": width, "height": height, "gt_boxes_xyxy": boxes}, annotation_bytes)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, (row, annotation_bytes) in enumerate(pool.map(read_voc, ids)):
                annotations_hash.update(row["id"].encode() + annotation_bytes)
                rows.append(row)
                if (i + 1) % 2000 == 0:
                    print("Read", name, i + 1, "/", len(ids), flush=True)
        datasets[name] = rows
        inputs[name] = {"split": str(layout), "split_sha256": digest(layout),
                        "annotations_sha256": annotations_hash.hexdigest()}
    subset = PROJECT / "evaluation/vendor/tokencut/coco_20k_filenames.txt"
    filenames = subset.read_text().splitlines()
    ids = [int(Path(f).stem.split("_")[-1]) for f in filenames]
    if len(ids) != EXPECTED["COCO20K"] or len(set(ids)) != len(ids):
        raise ValueError("Incorrect official COCO20K list")
    annotation_path = data / "coco/annotations/instances_train2014.json"
    print("Reading COCO2014 annotations and resolving the fixed COCO20K image list", flush=True)
    coco = json.loads(annotation_path.read_text())
    images = {image["id"]: image for image in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    rows = []
    image_directories = [data / "coco/images" / split for split in ("train2014", "train2017", "val2017")]
    available = {directory: {entry.name for entry in os.scandir(directory)}
                 for directory in image_directories if directory.is_dir()}
    for image_id, filename in zip(ids, filenames):
        image = images[image_id]
        original_filename = Path(filename).name
        if image["file_name"] != original_filename:
            raise ValueError("COCO2014 image identity mismatch")
        # COCO2017 redistributes the same JPEG images across train and val.
        # Use 2014 annotations/subset, resolving by original numeric image ID.
        candidates = [data / "coco/images/train2014" / original_filename,
                      data / "coco/images/train2017" / f"{image_id:012d}.jpg",
                      data / "coco/images/val2017" / f"{image_id:012d}.jpg"]
        path = next((p for p in candidates if p.name in available.get(p.parent, set())), None)
        if path is None:
            raise FileNotFoundError("Missing COCO20K image: " + filename)
        rows.append({"id": str(image_id), "original_filename": filename, "image": str(path),
                     "width": image["width"], "height": image["height"],
                     "gt_boxes_xyxy": coco_boxes(annotations[image_id])})
    datasets["COCO20K"] = rows
    inputs["COCO20K"] = {"annotations": str(annotation_path), "annotations_sha256": digest(annotation_path),
                          "subset_sha256": digest(subset), "image_mapping": "COCO2014 IDs, existing 2017 JPEGs"}
    for name, rows in datasets.items():
        image_hash = hashlib.sha256()
        def verify_image(row):
            path = Path(row["image"])
            contents = path.read_bytes()
            with Image.open(BytesIO(contents)) as image:
                if image.size != (row["width"], row["height"]):
                    raise ValueError("Image/annotation dimensions disagree: " + str(path))
                image.verify()
            # Decode the whole image as well: JPEG verify() alone does not check its stream.
            with Image.open(BytesIO(contents)) as image:
                image.convert("RGB").load()
            return row["id"].encode() + hashlib.sha256(contents).digest()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, image_digest in enumerate(pool.map(verify_image, rows)):
                image_hash.update(image_digest)
                if (i + 1) % 2000 == 0:
                    print("Validated", name, i + 1, "/", len(rows), flush=True)
        inputs[name]["images_sha256"] = image_hash.hexdigest()
        inputs[name]["images"] = len(rows)
        inputs[name]["images_without_noncrowd_boxes"] = sum(not r["gt_boxes_xyxy"] for r in rows)
    return datasets, inputs


def prepare(args):
    import yaml
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    for entry in config["models"].values():
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = config_path.parent / checkpoint
        entry["checkpoint"] = str(checkpoint.resolve(strict=True))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records, inputs = dataset_records(config)
    for name, rows in records.items():
        save(root / "datasets" / (name + ".json"), rows)
    source = root / "source"
    for directory in ("evaluation", "model", "utils"):
        for path in (PROJECT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".txt", ".md"):
                target = source / path.relative_to(PROJECT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    license_path = PROJECT / "evaluation/vendor/tokencut/LICENSE"
    shutil.copy2(license_path, source / license_path.relative_to(PROJECT))
    shutil.copy2(config_path, root / "config.yaml")
    shutil.copy2(PROJECT / "slurm/evaluation_object_discovery.sh", root / "evaluation.sh")
    tasks = [{"model": model, "dataset": name} for name in config["datasets"] for model in config["models"]]
    manifest = {
        "prepared_at": now(), "config": config, "data": inputs, "tasks": tasks, "sources": SOURCES,
        "checkpoint_sha256": {name: digest(entry["checkpoint"]) for name, entry in config["models"].items()},
        "dataset_manifest_sha256": {name: digest(root / "datasets" / (name + ".json")) for name in records},
        "source_sha256": {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()},
        "protocol_details": {
            "sweep_extension": "0.00 through 0.95 in steps of 0.05 for every model; published range 0.00 through 0.40 retained separately",
            "feature_layer": "final block, final LayerNorm, discard CLS/registers, no projection head",
            "precision": "float32, AMP/TF32 disabled", "image_normalization": "ImageNet mean/std",
            "native_resolution": "no resizing/cropping; zero-pad normalized pixels to next patch multiple",
            "graph": "official TokenCut ncut, binary cosine > tau, epsilon 1e-5, generalized dense eigensolve",
            "foreground": "eigenvector above mean; flip toward maximum absolute entry; its 4-connected component",
            "bbox": "component bounds times patch size, clipped to unpadded image extent",
            "metric": "percent images with any non-crowd GT box IoU > 0.5, also save >=0.5",
            "selection": "dataset-level best threshold is an oracle sweep summary using evaluation GT; every threshold is also reported; never choose a threshold per image",
            "limitations": "Meta object-discovery evaluator unpublished; unspecified details follow official TokenCut",
        },
    }
    save(root / "manifest.json", manifest)
    collect(args)
    print("Prepared", root, "tasks:", len(tasks), flush=True)


def summarize_rows(rows, total):
    from evaluation.utils.object_discovery import THRESHOLDS, PUBLISHED_THRESHOLDS
    scores = []
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        correct = sum(row[key]["max_iou"] > 0.5 for row in rows)
        correct_ge = sum(row[key]["max_iou"] >= 0.5 for row in rows)
        scores.append({"threshold": threshold, "correct": correct, "corloc": 100 * correct / len(rows) if rows else None,
                       "corloc_ge_0_5": 100 * correct_ge / len(rows) if rows else None})
    best = max(scores, key=lambda r: r["corloc"]) if rows else None
    return {"status": "completed" if len(rows) == total else "running", "evaluated_images": len(rows),
            "expected_images": total, "threshold_results": scores,
            "published_range_best": max((r for r in scores if r['threshold'] in PUBLISHED_THRESHOLDS), key=lambda r: r['corloc']) if rows and len(rows) == total else None,
            "best": best if len(rows) == total else None, "provisional_best": best, "updated_at": now()}


def open_database(path, identity):
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL)")
    stored = connection.execute("SELECT value FROM identity").fetchall()
    if stored and stored != [(identity,)]:
        connection.close()
        raise ValueError("Cannot resume predictions from a different checkpoint/protocol/dataset/source")
    if not stored:
        connection.execute("INSERT INTO identity VALUES (?)", (identity,))
    connection.execute("CREATE TABLE IF NOT EXISTS predictions (image_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.commit()
    return connection


def run(args):
    import torch
    import numpy as np
    import scipy
    import sys
    from PIL import __version__ as pillow_version
    from PIL import Image
    from evaluation.utils.common import load_backbone
    from evaluation.utils.object_discovery import native_image_tensor, predict_boxes, box_iou

    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    task = manifest["tasks"][args.task]
    entry = manifest["config"]["models"][task["model"]]
    dataset_path = root / "datasets" / (task["dataset"] + ".json")
    if digest(dataset_path) != manifest["dataset_manifest_sha256"][task["dataset"]]:
        raise ValueError("Dataset manifest changed after preparation")
    if digest(entry["checkpoint"]) != manifest["checkpoint_sha256"][task["model"]]:
        raise ValueError("Checkpoint changed after preparation")
    for relative, expected_hash in manifest["source_sha256"].items():
        if digest(PROJECT / relative) != expected_hash:
            raise ValueError("Frozen evaluation source changed: " + relative)
    records = json.loads(dataset_path.read_text())
    directory = root / task["model"] / task["dataset"]
    directory.mkdir(parents=True, exist_ok=True)
    identity = digest(root / "manifest.json") + ":" + str(args.task)
    connection = open_database(directory / "predictions.sqlite3", identity)
    completed = {image_id: json.loads(value) for image_id, value in connection.execute("SELECT image_id,value FROM predictions")}
    if set(completed) - {r["id"] for r in records}:
        raise ValueError("Predictions contain images outside the official dataset")
    if not torch.cuda.is_available():
        raise RuntimeError("Run full object discovery on an allocated GPU")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_backbone(entry["checkpoint"], entry["checkpoint_key"], "vit_small")
    if metadata["patch_size"] != 16:
        raise ValueError("Expected a ViT-S/16 checkpoint")
    model = model.float().cuda().eval()
    runtime = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
               "scipy": scipy.__version__, "pillow": pillow_version, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "cpu_threads": torch.get_num_threads()}
    print("Evaluating", task, metadata, "resume images", len(completed), flush=True)
    def report():
        result = summarize_rows(list(completed.values()), len(records))
        result.update(task=task, checkpoint=metadata, runtime=runtime, protocol=manifest["config"]["protocol"],
                      protocol_details=manifest["protocol_details"], manifest_sha256=digest(root / "manifest.json"))
        save(directory / "results.json", result)
        print(task, result["status"], len(completed), "/", len(records),
              "provisional best", result["provisional_best"], flush=True)
    report()
    with torch.inference_mode():
        for row in records:
            if row["id"] in completed:
                continue
            with Image.open(row["image"]) as image:
                tensor, original_size = native_image_tensor(image)
            if original_size != (row["height"], row["width"]):
                raise ValueError("Image dimensions changed after validation")
            tensor = tensor.unsqueeze(0).cuda()
            outputs = model.get_intermediate_layers(tensor, n=1)[0][:, 1:, :]
            dims = [tensor.shape[-2] // 16, tensor.shape[-1] // 16]
            predictions = predict_boxes(outputs, dims, original_size)
            for prediction in predictions.values():
                ious = box_iou(prediction["bbox_xyxy"], row["gt_boxes_xyxy"])
                prediction["max_iou"] = float(ious.max()) if len(ious) else 0.0
            # All thresholds for this image are committed together. Requeue repeats
            # only an interrupted image; previously committed predictions are reused.
            with connection:
                connection.execute("INSERT INTO predictions VALUES (?,?)", (row["id"], json.dumps(predictions)))
            completed[row["id"]] = predictions
            if len(completed) % 20 == 0:
                report()
    report()
    connection.close()


def collect(args):
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    results = {}
    for task in manifest["tasks"]:
        result_path = root / task["model"] / task["dataset"] / "results.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "pending"}
        results.setdefault(task["model"], {})[task["dataset"]] = result
    status = "completed" if all(r["status"] == "completed" for model in results.values() for r in model.values()) else "pending_or_running"
    save(root / "results_summary.json", {"status": status, "results": results, "sources": SOURCES, "updated_at": now()})
    print("Summary", status, root / "results_summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/evaluation_object_discovery.yaml")
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.command == "run" and args.task is None:
        parser.error("run requires --task")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
