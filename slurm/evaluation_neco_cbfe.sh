#!/usr/bin/env bash
#SBATCH --job-name=neco-cbfe-cd
#SBATCH --partition=3090,rtx8000,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=output/neco_cbfe_%A_%a.out
#SBATCH --error=output/neco_cbfe_%A_%a.err
set -euo pipefail
RUN_ROOT="${1:?Pass the run directory}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4 HF_HUB_OFFLINE=1
export PYTHONPATH="/mnt/fast/nobackup/scratch4weeks/jg02228/neco-cbfe-deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$RUN_ROOT/source"
srun --ntasks=1 --cpu-bind=cores \
  /mnt/fast/nobackup/scratch4weeks/jg02228/neco-eval-env/bin/python -u \
  -m evaluation.neco_benchmarks --config "$RUN_ROOT/config.yaml" \
  --root "$RUN_ROOT" --task "${SLURM_ARRAY_TASK_ID}"
