#!/usr/bin/env bash
# Prepare the 1024-pixel files expected by the released Probe3D NAVI reader.

#SBATCH --job-name=prepare-navi-eval
#SBATCH --partition=2080ti,3090,a100,rtx8000,rtx5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=output/prepare_navi_eval_%j.out
#SBATCH --error=output/prepare_navi_eval_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
./.conda-env/bin/python -m evaluation.prepare_navi \
    /mnt/fast/nobackup/scratch4weeks/jg02228/datasets/navi_v1 \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
