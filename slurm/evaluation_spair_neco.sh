#!/usr/bin/env bash
# Usage: sbatch --array=0-1 slurm/evaluation_spair_neco.sh PREPARED_RUN_ROOT
#SBATCH --job-name=spair-neco-224
#SBATCH --partition=rtx8000,3090,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output/spair_neco_%A_%a.out
#SBATCH --error=output/spair_neco_%A_%a.err
set -euo pipefail
RUN_ROOT="${1:?Pass the prepared evaluation run directory}"
MODELS=(ibot_original region200)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:?Submit with --array=0-1}]}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$RUN_ROOT"
srun --ntasks=1 --cpu-bind=cores \
    /mnt/fast/nobackup/scratch4weeks/jg02228/probe3d/env_probe3d/bin/python -u \
    "$RUN_ROOT/runner.py" run --root "$RUN_ROOT" --model "$MODEL"
