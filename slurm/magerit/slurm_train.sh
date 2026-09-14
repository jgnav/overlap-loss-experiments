#!/bin/bash -l
#SBATCH --job-name=ibot_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=standard-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:l40s:4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONFIG_PATH="${1:-train.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Training configuration not found: ${CONFIG_PATH}" >&2
    exit 2
fi

module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# shellcheck disable=SC1091
source .venv/bin/activate

export OMP_NUM_THREADS=1
export IBOT_RUN_ID="${IBOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-slurm-${SLURM_JOB_ID}}"

echo "Config:       ${CONFIG_PATH}"
echo "Output run:   ${IBOT_RUN_ID}"

srun torchrun --standalone --nproc-per-node=4 train.py "${CONFIG_PATH}"
