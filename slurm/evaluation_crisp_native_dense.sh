#!/usr/bin/env bash
# Usage: sbatch [resource overrides] slurm/evaluation_crisp_native_dense.sh RUN_ROOT TASK
# TASK: depth_nyu, depth_navi, snorm_nyu, snorm_navi
# Saved CRISP batch sizes: depth=2, surface normals=8, single GPU.
#SBATCH --job-name=crisp-dense-region200
#SBATCH --partition=rtx8000,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/crisp_dense_%j.out
#SBATCH --error=output/crisp_dense_%j.err
set -euo pipefail
RUN_ROOT="${1:?Pass the prepared evaluation run directory}"
TASK="${2:?Pass the benchmark task}"
PYTHON=/mnt/fast/nobackup/scratch4weeks/jg02228/probe3d/env_probe3d/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1 HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 --cpu-bind=cores "$PYTHON" -u "$RUN_ROOT/native_dense.py" \
    preflight --root "$RUN_ROOT" --task "$TASK"
srun --ntasks=1 --cpu-bind=cores "$PYTHON" -u "$RUN_ROOT/native_dense.py" \
    run --root "$RUN_ROOT" --task "$TASK"
# Every task writes its own metrics; the dependent collector builds the shared summary.
