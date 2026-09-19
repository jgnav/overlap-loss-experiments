#!/usr/bin/env bash

# Full resume of W&B run 9he0q7q7 from its latest preserved checkpoint.
# Training data remains in output/train/57317.
#SBATCH --job-name=train57317-resume
#SBATCH --partition=rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=output/train_resume_57317_%j.out
#SBATCH --error=output/train_resume_57317_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export IBOT_RUN_ID=57317
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
echo "Resuming checkpoint: output/train/57317/checkpoint_source0850_continuation0050.pth"
echo "W&B run: 9he0q7q7"
nvidia-smi

exec ./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py config/train_resume_57317_9he0q7q7.yaml
