#!/usr/bin/env bash

# 100 independent CPU jobs reuse the completed single-image PCA export.
# sbatch slurm/pca_color_variants.sh output/pca_visualizations_image_006_original
#SBATCH --job-name=pca-colors
#SBATCH --array=0-99%10
#SBATCH --partition=2080ti,debug
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/pca_colors_%A_%a.out
#SBATCH --error=logs/pca_colors_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${1:?Supply the completed single-image PCA output directory}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
printf -v variant '%03d' "$SLURM_ARRAY_TASK_ID"
srun ./.conda-env/bin/python -u pca_visualization.py \
    --recolor-from "$1" --color-variant "$SLURM_ARRAY_TASK_ID" \
    --output-dir "output/pca_color_variants_${SLURM_ARRAY_JOB_ID}/variant_${variant}"
