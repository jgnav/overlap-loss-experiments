#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=output/train/55493/slurm_%j.out
#SBATCH --error=output/train/55493/slurm_%j.err


set -e

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
# Reuse the existing run directory on every Slurm requeue, rather than making
# a new output/train/<new-job-id> directory.  The YAML resumes its checkpoint.
export IBOT_RUN_ID=55493

echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi

./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=6 \
    train.py config/train.yaml
