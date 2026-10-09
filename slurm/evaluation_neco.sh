#!/usr/bin/env bash
#SBATCH --job-name=eval-neco
#SBATCH --partition=rtx8000,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/neco_%A_%a.out
#SBATCH --error=output/neco_%A_%a.err
set -euo pipefail
RUN_ROOT="${1:?Pass the run directory}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4 HF_HUB_OFFLINE=1
cd "$RUN_ROOT/source"
srun --ntasks=1 --cpu-bind=cores \
  /mnt/fast/nobackup/scratch4weeks/jg02228/neco-eval-env/bin/python -u \
  -m evaluation.neco_benchmarks --config "$RUN_ROOT/config.yaml" \
  --root "$RUN_ROOT" --task "${SLURM_ARRAY_TASK_ID}"
