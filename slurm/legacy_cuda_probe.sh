#!/bin/bash
# Reproduce the CUDA portion of the pre-ablation launcher environment.

#SBATCH --job-name=ibot-legacy-cuda-probe
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/abaltion/legacy_cuda_probe_%j.out
#SBATCH --error=logs/abaltion/legacy_cuda_probe_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Node: $(hostname)"
nvidia-smi --query-gpu=uuid,pci.bus_id,driver_version --format=csv,noheader

./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    slurm/legacy_cuda_probe.py
