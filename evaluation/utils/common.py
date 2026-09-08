import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from model import create_model


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = ("vit_small", "vit_base", "vit_large")


def print_progress(description, current, total, updates=10):
    """Print bounded progress updates without terminal control characters."""
    if total <= 0:
        return
    interval = max(1, total // updates)
    if current == 1 or current == total or current % interval == 0:
        print(f"{description}: {current}/{total}", flush=True)


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("checkpoint", type=Path, help="iBOT/region-loss checkpoint")
    parser.add_argument(
        "--checkpoint-key",
        default="teacher",
        choices=("teacher", "student"),
        help="Network to evaluate from a full training checkpoint",
    )
    parser.add_argument(
        "--arch",
        default="auto",
        choices=("auto", *ARCHITECTURES),
        help="Infer the ViT size from the checkpoint by default",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="Directory containing the pre-downloaded datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evaluation root; defaults to a checkpoint-specific directory",
    )
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classification-manifests", type=Path, default=None,
        help="Multilabel split/label JSON directory; defaults to <datasets-root>/evaluation_manifests",
    )
    return parser


def prepare_paths(args, evaluation_name):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.datasets_root = args.datasets_root.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.datasets_root.is_dir():
        raise FileNotFoundError(
            f"Datasets directory not found: {args.datasets_root}"
        )
    if args.output_dir is None:
        checkpoint_name = re.sub(
            r"[^A-Za-z0-9_.-]+", "-", args.checkpoint.stem
        ).strip("-_")
        args.output_dir = (
            REPO_ROOT
            / "output"
            / "evaluation"
            / f"{checkpoint_name}-{checkpoint_fingerprint(args.checkpoint)}"
        )
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir = args.output_dir / evaluation_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.result_json is None:
        args.result_json = args.output_dir / "results.json"
    else:
        args.result_json = args.result_json.expanduser().resolve()
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
    return args


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def checkpoint_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    material = f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(material).hexdigest()[:16]


def classification_manifest_root(args):
    path = getattr(args, "classification_manifests", None)
    return (Path(path) if path is not None else Path(args.datasets_root) / "evaluation_manifests").expanduser().resolve()


def evaluation_identity(args):
    """Invalidate results when the evaluator, selected inputs, or seed changes."""
    digest = hashlib.sha256()
    for directory in (REPO_ROOT / "evaluation", REPO_ROOT / "model"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(str(path.relative_to(REPO_ROOT)).encode())
            digest.update(path.read_bytes())
    manifests = classification_manifest_root(args)
    manifest_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(manifests.glob("*.json"))
    }
    return {
        "version": 2,
        "source_sha256": digest.hexdigest(),
        "seed": args.seed,
        "architecture_argument": args.arch,
        "datasets_root": str(Path(args.datasets_root).expanduser().resolve()),
        "classification_manifests": str(manifests),
        "classification_manifest_hashes": manifest_hashes,
    }


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_state(checkpoint, checkpoint_key):
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state = checkpoint[checkpoint_key]
    elif (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("state_dict"), dict)
    ):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        state = checkpoint
    else:
        available = sorted(checkpoint) if isinstance(checkpoint, dict) else []
        raise ValueError(
            f"Checkpoint has no '{checkpoint_key}' weights. Available keys: "
            f"{available}"
        )
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint entry '{checkpoint_key}' is not a state dict")
    return state


def _canonical_backbone_state(state):
    canonical = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        while name.startswith("module.") or name.startswith("_orig_mod."):
            name = name.split(".", 1)[1]
        if name.startswith("backbone."):
            name = name[len("backbone.") :]
        canonical[name] = value
    return canonical


def _checkpoint_argument(checkpoint, name):
    if not isinstance(checkpoint, dict):
        return None
    arguments = checkpoint.get("args")
    if isinstance(arguments, dict):
        return arguments.get(name)
    return getattr(arguments, name, None)


def _infer_architecture(checkpoint, state):
    configured = _checkpoint_argument(checkpoint, "arch")
    if configured in ARCHITECTURES:
        return configured
    cls_token = state.get("cls_token")
    if cls_token is None:
        raise ValueError(
            "Cannot infer architecture: checkpoint has no backbone cls_token"
        )
    dimensions = {384: "vit_small", 768: "vit_base", 1024: "vit_large"}
    try:
        return dimensions[cls_token.shape[-1]]
    except KeyError as error:
        raise ValueError(
            f"Unsupported checkpoint hidden dimension: {cls_token.shape[-1]}"
        ) from error


def load_backbone(checkpoint_path, checkpoint_key="teacher", arch="auto"):
    checkpoint = _torch_load(checkpoint_path)
    raw_state = _checkpoint_state(checkpoint, checkpoint_key)
    state = _canonical_backbone_state(raw_state)
    architecture = _infer_architecture(checkpoint, state) if arch == "auto" else arch
    configured_arch = _checkpoint_argument(checkpoint, "arch")
    if configured_arch in ARCHITECTURES and configured_arch != architecture:
        raise ValueError(
            f"Requested architecture {architecture} does not match checkpoint "
            f"architecture {configured_arch}"
        )
    patch_weight = state.get("patch_embed.proj.weight")
    patch_size = (
        int(patch_weight.shape[-1])
        if patch_weight is not None
        else int(_checkpoint_argument(checkpoint, "patch_size") or 16)
    )
    if patch_size != 16:
        raise ValueError(
            f"CRISP evaluation requested ViT patch size 16, got {patch_size}"
        )
    model = create_model(
        architecture,
        patch_size=patch_size,
        num_classes=0,
        return_all_tokens=True,
    )
    model_keys = set(model.state_dict())
    filtered = {key: value for key, value in state.items() if key in model_keys}
    missing = sorted(model_keys - set(filtered))
    if missing:
        raise ValueError(
            "Checkpoint is missing backbone weights: " + ", ".join(missing[:10])
        )
    incompatible = model.load_state_dict(filtered, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Incompatible backbone checkpoint: {incompatible}")
    model.requires_grad_(False)
    model.eval()
    metadata = {
        "architecture": architecture,
        "patch_size": patch_size,
        "checkpoint_key": checkpoint_key,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint_path),
    }
    return model, metadata


def launch_distributed_if_needed(module, required_world_size):
    current_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if launched:
        if current_world_size != required_world_size:
            raise RuntimeError(
                f"This CRISP protocol requires {required_world_size} GPUs, but "
                f"torchrun launched {current_world_size} processes"
            )
        return
    if required_world_size == 1:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_world_size:
        raise RuntimeError(
            f"This CRISP protocol requires {required_world_size} visible GPUs; "
            f"PyTorch reports {torch.cuda.device_count()}"
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(required_world_size),
        "--module",
        module,
        *sys.argv[1:],
    ]
    raise SystemExit(
        subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    )


def initialize_distributed(seed=0, allow_tf32=False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=4),
        )
    elif not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires an NVIDIA GPU")
    rank = dist.get_rank() if dist.is_initialized() else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        # Keep the group alive until PyTorch's distributed exception hook has
        # reported the original error. The process exit will release it.
        if sys.exc_info()[0] is not None:
            return
        dist.barrier()
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def utc_now():
    return datetime.now(timezone.utc).isoformat()
