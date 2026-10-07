#!/usr/bin/env bash
# Download and verify original YouTube-VOS 2018 without allocating a GPU.
# Submit from the repository root. PREPARATION_SCRIPT may point to a frozen copy.
#SBATCH --job-name=prepare-ytvos2018
#SBATCH --partition=rtx5000,rtx8000,a100,l40s_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/prepare_ytvos2018_%j.out
#SBATCH --error=output/prepare_ytvos2018_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
datasets_root=/mnt/fast/nobackup/scratch4weeks/jg02228/datasets
preparation_script=${PREPARATION_SCRIPT:-evaluation/prepare_youtube_vos2018.py}
./.conda-env/bin/python -u "$preparation_script" \
    --datasets-root "$datasets_root" --workers 2 --phase validation \
    --bsdtar /mnt/fast/nobackup/users/jg02228/miniconda3/bin/bsdtar
exec ./.conda-env/bin/python -u "$preparation_script" \
    --datasets-root "$datasets_root" --workers 2 --phase all --download-retry-hours 36 \
    --bsdtar /mnt/fast/nobackup/users/jg02228/miniconda3/bin/bsdtar
