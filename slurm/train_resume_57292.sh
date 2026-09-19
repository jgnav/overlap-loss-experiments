#!/usr/bin/env bash

# Resume W&B run j0dtvfng in the original output directory output/train/57292.
# The checkpoint and YAML preserve the original training configuration.
#SBATCH --job-name=train57292-resume
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=output/train_resume_57292_%j.out
#SBATCH --error=output/train_resume_57292_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export IBOT_RUN_ID=57292
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
echo "Resuming checkpoint: output/train/57292/checkpoint.pth"
echo "W&B run: j0dtvfng"
nvidia-smi

exec ./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py config/train_resume_57292.yaml
