#!/bin/bash -l
#SBATCH --job-name=ibot_evaluation
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=standard-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONFIG_PATH="${1:-evaluation.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Evaluation configuration not found: ${CONFIG_PATH}" >&2
    exit 2
fi

module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# shellcheck disable=SC1091
source .venv/bin/activate

export OMP_NUM_THREADS=1

echo "Config:       ${CONFIG_PATH}"

srun --ntasks=1 --cpu-bind=cores python -u evaluation.py "${CONFIG_PATH}"
