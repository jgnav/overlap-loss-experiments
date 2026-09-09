import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.utils.common import (
    ARCHITECTURES,
    REPO_ROOT,
    checkpoint_fingerprint,
    classification_manifest_root,
    evaluation_identity,
    utc_now,
    write_json,
)


EVALUATIONS = (
    ("pascal_voc_knn", "evaluation.utils.pascal_voc_knn", None),
    ("pascal_voc_linear", "evaluation.utils.pascal_voc_linear", None),
    ("imagenet_knn", "evaluation.utils.imagenet_knn", None),
    ("imagenet_knn_1pct", "evaluation.utils.imagenet_knn_1pct", None),
    ("imagenet_knn_100pct", "evaluation.utils.imagenet_knn_100pct", None),
    ("ade20k_knn", "evaluation.utils.ade20k_knn", None),
    ("ade20k_linear", "evaluation.utils.ade20k_linear", None),
    ("cityscapes_knn", "evaluation.utils.cityscapes_knn", None),
    ("cityscapes_linear", "evaluation.utils.cityscapes_linear", None),
    ("imagenet_linear", "evaluation.utils.imagenet_linear", None),
    ("pascal_voc_multilabel", "evaluation.utils.pascal_voc_multilabel", None),
    ("pascal_voc_1shot", "evaluation.utils.pascal_voc_1shot", None),
    ("pascal_voc_2shot", "evaluation.utils.pascal_voc_2shot", None),
    ("pascal_voc_5shot", "evaluation.utils.pascal_voc_5shot", None),
    ("coco_multilabel", "evaluation.utils.coco_multilabel", None),
)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run segmentation and classification evaluations for one checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-key", default="teacher", choices=("teacher", "student")
    )
    parser.add_argument(
        "--arch", default="auto", choices=("auto", *ARCHITECTURES)
    )
    parser.add_argument(
        "--datasets-root", type=Path, default=REPO_ROOT / "dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory; defaults to a unique directory in output/evaluation",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Final table path; defaults to <output-dir>/full_evaluation.json",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classification-manifests", type=Path, default=None,
        help="Multilabel split/label JSON directory; defaults to <datasets-root>/evaluation_manifests",
    )
    parser.add_argument(
        "--evaluations", nargs="+", choices=[name for name, _, _ in EVALUATIONS],
        default=None, help="Run selected evaluations (default: all fifteen)",
    )
    return parser


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_") or "checkpoint"


def _result_table(results):
    table = []
    pairs = (
        ("PASCAL VOC 2012", "pascal_voc_knn", "pascal_voc_linear"),
        ("ImageNet-1K 1%", "imagenet_knn_1pct", None),
        ("ImageNet-1K 10%", "imagenet_knn", None),
        ("ImageNet-1K 100%", "imagenet_knn_100pct", None),
        ("ADE20K", "ade20k_knn", "ade20k_linear"),
        ("Cityscapes", "cityscapes_knn", "cityscapes_linear"),
    )
    for dataset, knn_name, linear_name in pairs:
        if knn_name not in results and (linear_name is None or linear_name not in results):
            continue
        row = {"dataset": dataset, "task": "multiclass_classification" if dataset.startswith("ImageNet") else "semantic_segmentation"}
        if knn_name in results:
            row["knn"] = results[knn_name].get("metrics", {})
        if linear_name is not None and linear_name in results:
            row["linear"] = results[linear_name].get("metrics", {})
        table.append(row)
    if "imagenet_linear" in results:
        table.append({
            "dataset": "ImageNet-1K 100%", "task": "multiclass_classification",
            "linear": results["imagenet_linear"]["metrics"],
        })
    for name in ("pascal_voc_1shot", "pascal_voc_2shot", "pascal_voc_5shot", "pascal_voc_multilabel", "coco_multilabel"):
        if name in results:
            result = results[name]
            table.append({
                "dataset": result["dataset"], "task": "multilabel_classification",
                "regime": name.removeprefix("pascal_voc_") if name.endswith("shot") else "full",
                "linear": result["metrics"],
            })
    return table


def _write_summary(path, args, started_at, status, results, error=None):
    model = None
    if results:
        model = next(iter(results.values())).get("model")
    updated_at = utc_now()
    summary = {
        "status": status,
        "checkpoint": str(args.checkpoint),
        "checkpoint_key": args.checkpoint_key,
        "architecture": args.arch,
        "model": model,
        "datasets_root": str(args.datasets_root),
        "evaluation_identity": evaluation_identity(args),
        "started_at": started_at,
        "updated_at": updated_at,
        "completed_evaluations": list(results),
        "evaluations": results,
        "table": _result_table(results),
    }
    if status in {"completed", "failed"}:
        summary["finished_at"] = updated_at
    if error is not None:
        summary["error"] = error
    write_json(path, summary)


def _load_completed_result(path, args, evaluation_name):
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    model = result.get("model", {})
    if (
        result.get("status") != "completed"
        or result.get("evaluation") != evaluation_name
        or model.get("checkpoint_fingerprint")
        != checkpoint_fingerprint(args.checkpoint)
        or model.get("checkpoint_key") != args.checkpoint_key
        or result.get("evaluation_identity") != evaluation_identity(args)
    ):
        return None
    return result


def _preflight_classification(args, evaluations):
    # Fail on missing/invalid annotation manifests before expensive segmentation
    # or 200-epoch classification work begins. No data is downloaded implicitly.
    from evaluation.utils.classification_data import (
        MULTILABEL_DATASETS, VOC_SHOT_EVALUATIONS, read_multilabel_manifest, sample_few_shot_indices,
    )

    names = {name for name, _, _ in evaluations}
    for dataset_name, spec in MULTILABEL_DATASETS.items():
        shot_counts = []
        if dataset_name == "pascal_voc":
            shot_counts = [shots for name, shots in VOC_SHOT_EVALUATIONS.items() if name in names]
        if f"{dataset_name}_multilabel" in names or shot_counts:
            samples, _, _ = read_multilabel_manifest(
                classification_manifest_root(args) / f"{dataset_name}.json",
                args.datasets_root, dataset_name, spec["num_classes"],
            )
            if shot_counts:
                sample_few_shot_indices([row[1] for row in samples["train"]], max(shot_counts), args.seed)


def main(argv=None):
    args = _parser().parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.datasets_root = args.datasets_root.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.datasets_root.is_dir():
        raise FileNotFoundError(f"Datasets directory not found: {args.datasets_root}")
    evaluations = [item for item in EVALUATIONS if args.evaluations is None or item[0] in args.evaluations]
    _preflight_classification(args, evaluations)
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = (
            REPO_ROOT
            / "output"
            / "evaluation"
            / f"{_safe_name(args.checkpoint.stem)}-{timestamp}"
        )
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.result_json is None:
        args.result_json = args.output_dir / "full_evaluation.json"
    else:
        args.result_json = args.result_json.expanduser().resolve()
    started_at = utc_now()
    results = {}
    _write_summary(
        args.result_json, args, started_at, "running", results
    )

    print(f"Starting {len(evaluations)} evaluations", flush=True)
    for evaluation_index, (name, module, cache_name) in enumerate(
        evaluations, start=1
    ):
        result_path = args.output_dir / f"{name}.json"
        completed_result = _load_completed_result(result_path, args, name)
        if completed_result is not None:
            results[name] = completed_result
            _write_summary(
                args.result_json, args, started_at, "running", results
            )
            print(
                f"[{evaluation_index}/{len(evaluations)}] "
                f"Reusing completed {name}",
                flush=True,
            )
            continue
        print(
            f"\n[{evaluation_index}/{len(evaluations)}] Starting {name}",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            module,
            str(args.checkpoint),
            "--checkpoint-key",
            args.checkpoint_key,
            "--arch",
            args.arch,
            "--datasets-root",
            str(args.datasets_root),
            "--output-dir",
            str(args.output_dir),
            "--result-json",
            str(result_path),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--classification-manifests",
            str(classification_manifest_root(args)),
        ]
        if cache_name is not None:
            cache_path = args.output_dir / "feature_cache" / f"{cache_name}.pth"
            command.extend(("--feature-cache", str(cache_path)))
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            error = f"{name} exited with status {completed.returncode}"
            _write_summary(
                args.result_json,
                args,
                started_at,
                "failed",
                results,
                error=error,
            )
            print(f"Full evaluation stopped: {error}", flush=True)
            return completed.returncode
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                results[name] = json.load(handle)
        except (OSError, json.JSONDecodeError) as error_object:
            error = f"{name} did not produce a valid result JSON: {error_object}"
            _write_summary(
                args.result_json,
                args,
                started_at,
                "failed",
                results,
                error=error,
            )
            print(f"Full evaluation stopped: {error}", flush=True)
            return 1
        _write_summary(
            args.result_json, args, started_at, "running", results
        )
        print(f"Completed {name}", flush=True)

    _write_summary(
        args.result_json, args, started_at, "completed", results
    )
    print(
        f"Full evaluation completed. Result table: {args.result_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
