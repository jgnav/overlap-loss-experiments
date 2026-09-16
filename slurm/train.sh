#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=output/train_%j.out
#SBATCH --error=output/train_%j.err


set -e

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi

./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py config/train.yaml
