#!/bin/bash -l
#SBATCH --job-name=pca-dense
#SBATCH --partition=standard-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pca-dense-%j.out
#SBATCH --error=logs/pca-dense-%j.err

set -euo pipefail

# Submit this job from the repository root.
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs
module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0
module load libffi/3.4.4-GCCcore-12.2.0
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun python -u pca_visualization.py