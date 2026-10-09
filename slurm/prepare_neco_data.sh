#!/usr/bin/env bash
# CPU-only annotation preparation; no GPU allocation.
#SBATCH --job-name=prepare-neco-labels
#SBATCH --partition=rtx8000,a100,3090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/neco_data_%A_%a.out
#SBATCH --error=output/neco_data_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
TASK=voc
if [[ "${SLURM_ARRAY_TASK_ID}" == 1 ]]; then TASK=coco; fi
cd "${SLURM_SUBMIT_DIR}"
srun .conda-env/bin/python -u evaluation/prepare_neco_data.py \
  --datasets /mnt/fast/nobackup/scratch4weeks/jg02228/datasets \
  --root /mnt/fast/nobackup/scratch4weeks/jg02228/datasets/neco_eval --task "$TASK"
