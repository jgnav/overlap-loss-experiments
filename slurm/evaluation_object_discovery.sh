#!/usr/bin/env bash
# One frozen model/dataset per array task; all twenty TokenCut thresholds per image.
# Usage: sbatch --array=0-8 script.sh /absolute/prepared/run/root
#SBATCH --job-name=eval-dinov3-tokencut
#SBATCH --partition=3090,3090_risk,a100,rtx8000,rtx_a6000_risk,l40s_risk,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=output/tokencut_%A_%a.out
#SBATCH --error=output/tokencut_%A_%a.err
#SBATCH --open-mode=append
set -euo pipefail
RUN_ROOT="${1:?Pass a prepared object-discovery run directory}"
PYTHON="${SLURM_SUBMIT_DIR}/.conda-env/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$RUN_ROOT/source"
requeue_before_timeout() {
    trap '' USR1
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_before_timeout USR1
srun --ntasks=1 --cpu-bind=cores "$PYTHON" -u -m evaluation.object_discovery run \
    --root "$RUN_ROOT" --task "${SLURM_ARRAY_TASK_ID:?Submit as an array}" &
wait "$!"
