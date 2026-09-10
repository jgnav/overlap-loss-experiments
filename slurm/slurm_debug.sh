#!/bin/bash -l
#SBATCH --job-name=ibot_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=debug-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:v100:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# shellcheck disable=SC1091
source .venv/bin/activate

export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export IBOT_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-slurm-${SLURM_JOB_ID}"
# V100 does not support BF16. Keep the production YAML unchanged and run this
# resource-limited smoke test in FP32 with a smaller per-GPU batch.
export IBOT_PRECISION_OVERRIDE=fp32
export IBOT_BATCH_SIZE_PER_GPU_OVERRIDE=4
export IBOT_GPU_COUNT_OVERRIDE=1

srun torchrun --standalone --nproc-per-node=1 train.py train.yaml
