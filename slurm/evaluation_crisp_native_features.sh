#!/usr/bin/env bash
# Submit with --array=0-5 and a prepared run directory.
#SBATCH --job-name=crisp-region200-features
#SBATCH --partition=rtx8000,3090,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output/crisp_features_%A_%a.out
#SBATCH --error=output/crisp_features_%A_%a.err
set -euo pipefail
RUN_ROOT="${1:?Pass the prepared evaluation run directory}"
CRISP_ROOT=/mnt/fast/nobackup/scratch4weeks/jg02228/probe3d
VARIANTS=(final_norm last4_norm_mean)
DATASETS=(spair navi scannet)
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-5}"
VARIANT="${VARIANTS[$((TASK_ID / 3))]}"
DATASET="${DATASETS[$((TASK_ID % 3))]}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$RUN_ROOT/source"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 --cpu-bind=cores "$CRISP_ROOT/env_probe3d/bin/python" -u \
    "$RUN_ROOT/runner.py" run --root "$RUN_ROOT" --variant "$VARIANT" --dataset "$DATASET"
