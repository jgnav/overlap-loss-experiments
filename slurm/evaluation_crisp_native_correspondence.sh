#!/usr/bin/env bash
# Usage: sbatch --array=0-2 slurm/evaluation_crisp_native_correspondence.sh RUN_ROOT
# RUN_ROOT contains a frozen copy of the supplied CRISP code and dataset links.
#SBATCH --job-name=crisp-native-region200
#SBATCH --partition=rtx8000,3090,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output/crisp_native_%A_%a.out
#SBATCH --error=output/crisp_native_%A_%a.err
set -euo pipefail
RUN_ROOT="${1:?Pass the prepared evaluation run directory}"
CRISP_ROOT=/mnt/fast/nobackup/scratch4weeks/jg02228/probe3d
DATASETS=(spair navi scannet)
DATASET="${DATASETS[${SLURM_ARRAY_TASK_ID:?Submit with --array=0-2}]}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$RUN_ROOT/source"
test -f "$RUN_ROOT/preflight_manifest.json"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 --cpu-bind=cores "$CRISP_ROOT/env_probe3d/bin/python" -u \
    run_native.py "$DATASET" "$RUN_ROOT"
