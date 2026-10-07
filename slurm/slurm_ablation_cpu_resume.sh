#!/usr/bin/env bash
# Launch a prepared ablation config, or resume its original run ID/log paths.
# Fresh run: sbatch slurm/slurm_ablation_cpu_resume.sh <config>
# sbatch --output=<original.out> --error=<original.err> \
#   slurm/slurm_ablation_cpu_resume.sh <config> <run_id> <sync_probes:0|1>
#SBATCH --job-name=ablation-cpu-resume
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=50:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/ablation_cpu_resume_%j.out
#SBATCH --error=logs/ablation_cpu_resume_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit with sbatch}"
: "${SLURM_JOB_ID:?Missing Slurm job ID}"
cd "$SLURM_SUBMIT_DIR"
config_path="${1:?Pass the prepared training config}"
export IBOT_RUN_ID="${2:-$SLURM_JOB_ID}"
sync_probes="${3:-0}"
[[ "$IBOT_RUN_ID" =~ ^[0-9]+(_[0-9]+)?$ ]] || exit 2
[[ "$sync_probes" == 0 || "$sync_probes" == 1 ]] || exit 2
[[ -s "$config_path" ]] || exit 2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "$sync_probes" == 1 ]]; then
    export IBOT_SYNC_PROBES=1
else
    unset IBOT_SYNC_PROBES
fi
echo "Ablation $IBOT_RUN_ID: config=$config_path CPUs=${SLURM_CPUS_PER_TASK} restart=${SLURM_RESTART_COUNT:-0}"
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
