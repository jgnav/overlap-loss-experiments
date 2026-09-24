#!/usr/bin/env bash
# Resume array 70768 tasks 6 or 7 with synchronous probes in one training process.
# Usage: sbatch slurm/region_aggregation_resume_paused_probes.sh <6|7>

#SBATCH --job-name=ibot-region-aggregation-probes
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey14
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/abaltion/resume_probes_%j.out
#SBATCH --error=logs/abaltion/resume_probes_%j.err

set -euo pipefail

task="${1:?Specify original task ID: 6 or 7}"
case "$task" in
    6) name=region_aggregation_swd ;;
    7) name=region_aggregation_mean_centered_swd ;;
    *) echo "Only array 70768 tasks 6 and 7 are supported" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR}"
source slurm/training_runtime.sh
configure_training_compatibility
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export IBOT_RUN_ID="70768_${task}"
unset IBOT_PRECISION_OVERRIDE IBOT_BATCH_SIZE_PER_GPU_OVERRIDE IBOT_GPU_COUNT_OVERRIDE

run_dir="output/ablation/${IBOT_RUN_ID}"
[[ -s "${run_dir}/checkpoint.pth" ]] || { echo "Missing checkpoint in ${run_dir}" >&2; exit 2; }
compgen -G "${run_dir}/wandb/run-*" >/dev/null || { echo "Missing W&B run in ${run_dir}" >&2; exit 2; }

echo "Resuming ${IBOT_RUN_ID} with stopped-training online probes on $(hostname)"
runtime_config="${run_dir}/runtime_config.yaml"
prepare_training_config ./.conda-env/bin/python \
    "config/ablations/${name}.yaml" "$runtime_config" "$run_dir"
export IBOT_SYNC_PROBES=1
exec ./.conda-env/bin/torchrun --standalone --nproc_per_node=4 \
    train.py "$runtime_config"
