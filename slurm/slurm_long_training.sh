#!/usr/bin/env bash
# Submit small runs (tasks 0 and 3) on 3090, A100, or RTX Pro 6000:
#   sbatch --array=0,3 slurm/slurm_long_training.sh
# Submit base and large runs (tasks 1 and 2), including RTX 3090 GPUs:
#   sbatch --array=1-2 --partition=3090,3090_risk,a100,rtx_pro6000_risk slurm/slurm_long_training.sh

#SBATCH --job-name=ibot-long
#SBATCH --array=0-3
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/long_training_%A_%a.out
#SBATCH --error=logs/long_training_%A_%a.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
: "${SLURM_ARRAY_JOB_ID:?This launcher requires a Slurm job array}"
: "${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"
cd "$SLURM_SUBMIT_DIR"

configs=(
    ibot_vit_small
    ibot_vit_base
    ibot_vit_large
    ibot_vit_small_reference
)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#configs[@]} )); then
    echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
config_path="config/long_training/${configs[$SLURM_ARRAY_TASK_ID]}.yaml"
if [[ -n "${IBOT_RETRY_ORIGINAL_ARRAY_ID:-}" ]]; then
    [[ "$IBOT_RETRY_ORIGINAL_ARRAY_ID" =~ ^[0-9]+$ ]] || {
        echo "IBOT_RETRY_ORIGINAL_ARRAY_ID must be numeric" >&2
        exit 2
    }
    config_path="output/long_${configs[$SLURM_ARRAY_TASK_ID]}/${IBOT_RETRY_ORIGINAL_ARRAY_ID}_${SLURM_ARRAY_TASK_ID}/retry_config.yaml"
fi
if [[ ! -f "$config_path" ]]; then
    echo "Training configuration not found: $config_path" >&2
    exit 2
fi

export IBOT_RUN_ID="${IBOT_RETRY_ORIGINAL_ARRAY_ID:-$SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export IBOT_SYNC_PROBES=1

echo "Config: $config_path"
echo "Run ID: $IBOT_RUN_ID"
echo "Node: $(hostname)"
nvidia-smi

if [[ -x ./.conda-env/bin/torchrun ]]; then
    torchrun_bin=./.conda-env/bin/torchrun
elif [[ -x ./.venv/bin/torchrun ]]; then
    torchrun_bin=./.venv/bin/torchrun
else
    torchrun_bin=$(command -v torchrun) || {
        echo "torchrun not found; activate the training environment first" >&2
        exit 2
    }
fi

requeue_before_timeout() {
    trap - USR1
    echo "Approaching the Slurm time limit; requeueing ${SLURM_JOB_ID} for checkpoint resume"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_before_timeout USR1

"$torchrun_bin" \
    --standalone \
    --nproc_per_node=8 \
    train.py "$config_path" &
train_pid=$!
wait "$train_pid"
