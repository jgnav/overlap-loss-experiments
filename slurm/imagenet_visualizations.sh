#!/usr/bin/env bash

#SBATCH --job-name=imagenet-viz
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/imagenet_visualizations_%j.out
#SBATCH --error=logs/imagenet_visualizations_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/imagenet-visualizations-mpl"

echo "Node: $(hostname)"
nvidia-smi
exec ./.conda-env/bin/python -u imagenet_visualizations.py
