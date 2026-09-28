"""Prepare full-resume configs for interrupted ablations.

Each config keeps the original run directory and W&B ID. The checkpoint path
stays fixed, so a Slurm requeue reads the newest saved epoch on every start.
"""

import os
from pathlib import Path
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import load_config
from utils.checkpoint import _validate_resume_compatibility


RUNS = {
    "83650_0": "lambda3_0p4",
    "83650_3": "region_aggregation_mean_covariance",
    "83650_7": "region_normalization_softmax",
    "83657_10": "lambda3_0p2",
    "83657_12": "lambda3_0p8",
    "83657_15": "koleo_regularizer_true",
    "83657_17": "region_min_area_0p3",
    "83657_18": "region_min_area_0p5",
    "83657_19": "region_patch_threshold_0p2",
    "83657_20": "region_patch_threshold_0p5",
    "83657_21": "region_patch_threshold_weighted",
    "83657_22": "register_4",
    "83657_23": "shared_head_false",
}


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    requested = sys.argv[1:] or list(RUNS)
    unknown = set(requested) - set(RUNS)
    if unknown:
        raise ValueError(f"Unknown ablation run IDs: {sorted(unknown)}")
    for run_id in requested:
        config_name = RUNS[run_id]
        run_dir = root / "output" / "ablation" / run_id
        checkpoint_path = run_dir / "checkpoint.pth"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        epoch = checkpoint.get("epoch")
        if not isinstance(epoch, int) or not 0 < epoch < 50:
            raise ValueError(f"{run_id}: unexpected checkpoint epoch {epoch!r}")
        saved_args = checkpoint.get("args")
        if getattr(saved_args, "run_id", None) != run_id:
            raise ValueError(f"{run_id}: checkpoint belongs to another run")
        wandb_ids = {
            path.name.rsplit("-", 1)[-1]
            for path in (run_dir / "wandb").glob("run-*")
            if path.is_dir()
        }
        if len(wandb_ids) != 1:
            raise ValueError(f"{run_id}: expected one original W&B ID, got {wandb_ids}")
        wandb_id = wandb_ids.pop()
        config_path = root / "config" / "ablations" / f"{config_name}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["resume_checkpoint"] = str(checkpoint_path)
        config["reset_optimizer"] = False
        config["wandb_run_id"] = wandb_id
        config["wandb_resume"] = "must"
        output = run_dir / "resume_config.yaml"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=run_dir, prefix=".resume_config.",
            suffix=".yaml", delete=False,
        ) as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
            temporary = Path(handle.name)
        try:
            args = load_config(temporary)
            args.world_size = config["gpu_count"]
            args.effective_batch_size = args.batch_size_per_gpu * args.world_size
            _validate_resume_compatibility(checkpoint, args)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{run_id}: epoch {epoch}/50, W&B {wandb_id}, {output}", flush=True)
        del checkpoint


if __name__ == "__main__":
    sys.exit(main())
