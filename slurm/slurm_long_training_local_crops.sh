#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch slurm/slurm_long_training_local_crops.sh

#SBATCH --job-name=ibot-long-local-crops
#SBATCH --partition=a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --nice=1000
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/long_training_local_crops_%j.out
#SBATCH --error=logs/long_training_local_crops_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
: "${SLURM_JOB_ID:?Missing Slurm job ID}"
cd "$SLURM_SUBMIT_DIR"

config_path="config/long_training/ibot_vit_small_local_crops.yaml"
[[ -f "$config_path" ]] || { echo "Missing config: $config_path" >&2; exit 2; }
export IBOT_RUN_ID="$SLURM_JOB_ID"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export IBOT_SYNC_PROBES=1

echo "Config: $config_path"
echo "Run ID: $IBOT_RUN_ID"
echo "Node: $(hostname)"
nvidia-smi

requeue_before_timeout() {
    trap - USR1
    echo "Approaching Slurm time limit; requeueing $SLURM_JOB_ID for full resume"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_timeout USR1

./.conda-env/bin/torchrun --standalone --nproc_per_node=4 train.py "$config_path" &
train_pid=$!
wait "$train_pid"
