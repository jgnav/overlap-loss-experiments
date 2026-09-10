"""Prepare the active Python environment for evaluation worker processes."""

import os
from pathlib import Path
import sysconfig


def worker_environment():
    env = os.environ.copy()
    # RAPIDS/cuML loads these dynamically. Discover the active environment's
    # CUDA wheels rather than hardcoding a Python version in the Slurm script.
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    libraries = [str(path) for path in sorted(nvidia.glob("*/lib")) if path.is_dir()]
    if libraries:
        env["LD_LIBRARY_PATH"] = ":".join(libraries + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    return env
