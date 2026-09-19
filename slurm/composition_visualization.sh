#!/usr/bin/env bash

#SBATCH --job-name=composition-viz
#SBATCH --partition=debug,3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/composition_%j.out
#SBATCH --error=logs/composition_%j.err

set -euo pipefail

# Submit from the repository root. Inputs and output are configured in Python.
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
nvidia-smi

exec ./.conda-env/bin/python -u composition_visualization.py
