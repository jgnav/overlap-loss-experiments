#!/bin/bash
set -e

cd "$(dirname "$0")/.."

rm -rf .conda-env

conda create -y -p ./.conda-env python=3.11 pip

./.conda-env/bin/python -m pip install --upgrade pip
./.conda-env/bin/python -m pip install -r requirements.txt

./.conda-env/bin/python --version
./.conda-env/bin/python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"