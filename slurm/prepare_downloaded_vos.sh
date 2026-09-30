#!/usr/bin/env bash
# Extract the user-downloaded YouTube-VOS 2019 and MOSEv2 data on a CPU allocation.
#SBATCH --job-name=prepare-downloaded-vos
#SBATCH --partition=2080ti,3090,a100,rtx8000,rtx5000,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output/prepare_downloaded_vos_%j.out
#SBATCH --error=output/prepare_downloaded_vos_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"
export PATH="/mnt/fast/nobackup/users/jg02228/miniconda3/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
./.conda-env/bin/python -m evaluation.prepare_downloaded_vos \
    --datasets-root /mnt/fast/nobackup/scratch4weeks/jg02228/datasets
