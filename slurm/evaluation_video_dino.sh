#!/usr/bin/env bash
# sbatch slurm/evaluation_video_dino.sh config/evaluation_video_dino.yaml
# Arrays take a directory containing <array task ID>.yaml configurations.
#SBATCH --job-name=eval-video-dino
#SBATCH --partition=rtx8000,l40s_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=output/video_dino_%A_%a.out
#SBATCH --error=output/video_dino_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG_PATH="${1:-config/evaluation_video_dino.yaml}"
if [[ -d "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$CONFIG_PATH/${SLURM_ARRAY_TASK_ID:?array ID required}.yaml"
fi
[[ -f "$CONFIG_PATH" ]] || { echo "Missing config: $CONFIG_PATH" >&2; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
requeue_before_timeout() {
    trap '' USR1
    echo "Requeueing $SLURM_JOB_ID; completed videos will be reused."
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_timeout USR1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 --cpu-bind=cores ./.conda-env/bin/python -u evaluation.py "$CONFIG_PATH" &
wait "$!"
