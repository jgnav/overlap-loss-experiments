#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --ntasks=1
#SBATCH --constraint=fs_weka
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=100:00:00
#SBATCH --output=output/train_%j.out
#SBATCH --error=output/train_%j.err


set -e

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi

./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=6 \
    train.py