"""Run individually selected evaluations from one YAML configuration."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evaluation.utils.common import REPO_ROOT, classification_manifest_root, utc_now
from evaluation.utils.config import config_snapshot, load_config
from evaluation.utils.runtime import worker_environment
from utils.wandb_logging import init_wandb_run, log_evaluation
from evaluation.utils.orchestrator import (
    EVALUATIONS, _load_completed_result, _preflight_classification, _safe_name, _write_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=REPO_ROOT / "evaluation.yaml",
                        help="YAML configuration (default: evaluation.yaml beside this script)")
    return parser.parse_args(argv)


def evaluation_command(args, name, module, result_path):
    command = [
        sys.executable, "-m", module, str(args.checkpoint),
        "--checkpoint-key", args.checkpoint_key, "--arch", args.arch,
        "--datasets-root", str(args.datasets_root), "--output-dir", str(args.output_dir),
        "--result-json", str(result_path), "--num-workers", str(args.num_workers),
        "--seed", str(args.seed), "--classification-manifests", str(args.classification_manifests),
    ]
    if name in {"pascal_voc_knn", "pascal_voc_linear", "ade20k_knn", "ade20k_linear",
                "cityscapes_knn", "cityscapes_linear"}:
        command.extend(("--batch-size", str(args.segmentation_batch_size)))
    return command


def run_evaluations(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.datasets_root.is_dir():
        raise FileNotFoundError(f"Datasets directory not found: {args.datasets_root}")
    evaluations = [item for item in EVALUATIONS if item[0] in args.evaluations]
    args.classification_manifests = classification_manifest_root(args)
    _preflight_classification(args, evaluations)
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = REPO_ROOT / "output" / "evaluation" / f"{_safe_name(args.checkpoint.stem)}-{timestamp}"
    if args.result_json is None:
        args.result_json = args.output_dir / "full_evaluation.json"
    # Summary/config paths must not replace inputs or worker reports.
    worker_paths = {args.output_dir / f"{name}.json" for name, _, _ in EVALUATIONS}
    if (args.result_json in worker_paths or args.result_json in {args.checkpoint, args.config_path}
            or args.result_json == args.output_dir / "evaluation_config.yaml"):
        raise ValueError("result_json must differ from the checkpoint, config, and individual result paths")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = config_snapshot(args)
    snapshot_path = args.output_dir / "evaluation_config.yaml"
    if snapshot_path == args.checkpoint:
        raise ValueError("Output configuration snapshot must not overwrite the checkpoint")
    if snapshot_path != args.config_path:
        snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    started_at = utc_now()
    results = {}
    _write_summary(args.result_json, args, started_at, "running", results)
    print(f"Config: {args.config_path}\nCheckpoint: {args.checkpoint}\nDatasets: {args.datasets_root}", flush=True)
    print(f"Output: {args.output_dir}\nSelected: {', '.join(args.evaluations)}", flush=True)
    wandb_run = init_wandb_run(args, snapshot, "evaluation")
    exit_code = 1
    try:
        for index, (name, module, _) in enumerate(evaluations, start=1):
            result_path = args.output_dir / f"{name}.json"
            result = _load_completed_result(result_path, args, name)
            if result is not None:
                print(f"[{index}/{len(evaluations)}] Reusing completed {name}", flush=True)
            else:
                print(f"[{index}/{len(evaluations)}] Starting {name}", flush=True)
                try:
                    completed = subprocess.run(evaluation_command(args, name, module, result_path),
                                               cwd=REPO_ROOT, env=worker_environment(), check=False)
                    if completed.returncode != 0:
                        raise RuntimeError(f"{name} exited with status {completed.returncode}")
                    result = _load_completed_result(result_path, args, name)
                    if result is None:
                        raise RuntimeError(f"{name} did not produce a compatible completed result JSON")
                except (OSError, RuntimeError) as error:
                    _write_summary(args.result_json, args, started_at, "failed", results, error=str(error))
                    print(f"Evaluation stopped: {error}", flush=True)
                    return 1
            results[name] = result
            log_evaluation(wandb_run, name, result)
            _write_summary(args.result_json, args, started_at, "running", results)
        _write_summary(args.result_json, args, started_at, "completed", results)
        print(f"Evaluation completed. Result table: {args.result_json}", flush=True)
        exit_code = 0
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.summary["state/status"] = "completed" if exit_code == 0 else "failed"
            wandb_run.summary["state/completed_evaluations"] = list(results)
            wandb_run.summary["state/result_json"] = str(args.result_json)
            wandb_run.finish(exit_code=exit_code)


def main(argv=None):
    return run_evaluations(load_config(parse_args(argv).config))


if __name__ == "__main__":
    raise SystemExit(main())
