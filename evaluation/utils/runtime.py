"""Prepare the active Python environment for evaluation worker processes."""

import os
from pathlib import Path
import sysconfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_cudart_compatibility():
    """Expose PyTorch's versioned CUDA runtime under cuML's lookup name."""
    cuda_runtime = Path(sysconfig.get_path("purelib")) / "nvidia" / "cuda_runtime" / "lib"
    candidates = sorted(cuda_runtime.glob("libcudart.so.*"))
    if not candidates:
        return None
    source = candidates[-1].resolve()

    compatibility_dir = REPO_ROOT / ".runtime-libs"
    compatibility_dir.mkdir(parents=True, exist_ok=True)
    compatibility_link = compatibility_dir / "libcudart.so"
    if compatibility_link.is_symlink():
        if compatibility_link.resolve() != source:
            compatibility_link.unlink()
    elif compatibility_link.exists():
        raise RuntimeError(
            f"CUDA runtime compatibility path is not a symlink: {compatibility_link}"
        )
    if not compatibility_link.exists():
        try:
            compatibility_link.symlink_to(source)
        except FileExistsError:
            # Another evaluator may have created the same link concurrently.
            if not compatibility_link.is_symlink() or compatibility_link.resolve() != source:
                raise
    return compatibility_dir


def worker_environment():
    env = os.environ.copy()
    # RAPIDS/cuML loads these dynamically. Discover the active environment's
    # CUDA wheels rather than hardcoding a Python version in the Slurm script.
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    libraries = [str(path) for path in sorted(nvidia.glob("*/lib")) if path.is_dir()]
    compatibility_dir = _ensure_cudart_compatibility()
    if compatibility_dir is not None:
        libraries.insert(0, str(compatibility_dir))
    if libraries:
        env["LD_LIBRARY_PATH"] = ":".join(libraries + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    return env
