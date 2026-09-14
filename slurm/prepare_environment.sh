#!/bin/bash

#SBATCH --job-name=prepare_data
#SBATCH --partition=2080ti,3090,a100,rtx8000,rtx5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output/prepare_data_%j.out
#SBATCH --error=output/prepare_data_%j.err

set -e

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

./.conda-env/bin/python prepare_data.py \
    /mnt/fast/nobackup/scratch4weeks/jg02228/datasets