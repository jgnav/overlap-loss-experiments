#!/usr/bin/env bash
# Usage: sbatch slurm/evaluation_correspondence.sh CONFIG.yaml
# An array can instead receive a directory containing <array-task-id>.yaml.
#SBATCH --job-name=eval-correspondence
#SBATCH --partition=rtx8000,3090,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output/correspondence_%A_%a.out
#SBATCH --error=output/correspondence_%A_%a.err
set -euo pipefail
cd "${EVALUATION_SOURCE_ROOT:-${SLURM_SUBMIT_DIR}}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
CONFIG_PATH="${1:-config/evaluation_correspondence_region_vits200.yaml}"
if [[ -d "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$CONFIG_PATH/${SLURM_ARRAY_TASK_ID:?array task ID required}.yaml"
fi
test -f "$CONFIG_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 --cpu-bind=cores ./.conda-env/bin/python -u evaluation.py "$CONFIG_PATH"
