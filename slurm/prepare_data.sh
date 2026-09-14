#!/bin/bash

#SBATCH --job-name=prepare_data
#SBATCH --partition=2080ti,3090,a100,rtx8000,rtx5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --output=prepare_data_%j.out
#SBATCH --error=prepare_data_%j.err

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 prepare_data.py /mnt/fast/nobackup/scratch4weeks/jg02228/datasets
