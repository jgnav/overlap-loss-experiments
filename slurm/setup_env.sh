#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf .conda-env

conda create -y -p ./.conda-env python=3.11 pip

PYTHON="./.conda-env/bin/python"

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -r requirements.txt

"${PYTHON}" --version
"${PYTHON}" - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
arch_flags = torch._C._cuda_getArchFlags() or ""
arch_list = arch_flags.split()
print("compiled CUDA architectures:", arch_list)

if "sm_120" not in arch_list:
    raise SystemExit(
        "This PyTorch build does not contain sm_120 kernels required by "
        "the RTX PRO 6000 Blackwell GPUs."
    )
PY
