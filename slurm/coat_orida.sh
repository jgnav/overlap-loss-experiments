#!/usr/bin/env bash
# Run COAT-style compositionality evaluation on ORIDa using one GPU.
# Submit from the repository root with: sbatch slurm/coat_orida.sh

#SBATCH --job-name=coat-orida
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/coat_orida_%j.out
#SBATCH --error=logs/coat_orida_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
echo "ORIDa: /mnt/fast/nobackup/scratch4weeks/jg02228/datasets/orida/ORIDa_v1.0"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec srun --ntasks=1 --cpu-bind=cores ./.conda-env/bin/python -u coat_orida.py
