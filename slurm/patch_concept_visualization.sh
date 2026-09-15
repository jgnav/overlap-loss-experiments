#!/bin/bash

#SBATCH --job-name=patch-concepts
#SBATCH --partition=debug,3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/patch_concepts_%j.out
#SBATCH --error=logs/patch_concepts_%j.err

set -euo pipefail

# Submit from the repository root. Input/output paths are configured in Python.
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
nvidia-smi

exec ./.conda-env/bin/python -u patch_concept_visualization.py
