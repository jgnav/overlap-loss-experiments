#!/usr/bin/env bash
# Fresh B/L continuations with four GPUs and gradient accumulation.
# sbatch slurm/slurm_long_training_base_large.sh
# Array task 0: ViT-B, task 1: ViT-L.
#SBATCH --job-name=ibot-long-bl
#SBATCH --array=0-1
#SBATCH --partition=a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/long_training_bl_%A_%a.out
#SBATCH --error=logs/long_training_bl_%A_%a.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit with sbatch}"
: "${SLURM_ARRAY_JOB_ID:?This launcher requires a job array}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task ID}"
cd "$SLURM_SUBMIT_DIR"
configs=(ibot_vit_base ibot_vit_large)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#configs[@]} )); then
    echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
export IBOT_RUN_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 IBOT_SYNC_PROBES=1
config_path="config/long_training/${configs[$SLURM_ARRAY_TASK_ID]}.yaml"
echo "Training $config_path; run $IBOT_RUN_ID; Slurm restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi
requeue_before_timeout() {
    trap - USR1
    echo "Requeueing $SLURM_JOB_ID before timeout; next launch restores the latest checkpoint"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_timeout USR1
./.conda-env/bin/torchrun --standalone --nproc_per_node=4 train.py "$config_path" &
train_pid=$!
wait "$train_pid"
