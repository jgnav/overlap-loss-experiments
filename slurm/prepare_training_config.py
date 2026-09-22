#!/usr/bin/env python3
"""Create a per-allocation training YAML with safe checkpoint/W&B recovery."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import yaml


def wandb_run_id(run_dir: Path) -> str | None:
    candidates = sorted(
        (path for path in (run_dir / "wandb").glob("run-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        run_id = candidate.name.rsplit("-", 1)[-1]
        if run_id:
            return run_id
    return None


def prepare_config(base_config: Path, output_config: Path, run_dir: Path) -> dict:
    with base_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{base_config} must contain a YAML mapping")

    checkpoint = run_dir / "checkpoint.pth"
    if checkpoint.is_file():
        config["resume_checkpoint"] = str(checkpoint)
        config["reset_optimizer"] = False

    existing_run_id = config.get("wandb_run_id") or wandb_run_id(run_dir)
    if existing_run_id:
        config["wandb_run_id"] = existing_run_id
        # `allow` resumes an existing remote run and creates it when an
        # interrupted first allocation never reached the W&B service.
        config["wandb_resume"] = "allow"

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_config.parent,
        prefix=f".{output_config.name}.", delete=False,
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        temporary = Path(handle.name)
    os.replace(temporary, output_config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_config", type=Path)
    parser.add_argument("output_config", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    config = prepare_config(args.base_config, args.output_config, args.run_dir)
    checkpoint = config.get("resume_checkpoint")
    print(f"Resume checkpoint: {checkpoint if checkpoint else 'none'}")
    print(f"W&B run ID: {config.get('wandb_run_id') or 'new run'}")


if __name__ == "__main__":
    main()
