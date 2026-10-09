#!/usr/bin/env bash
# Four GPUs, 256 images/GPU. Multilabel selects iBOT, BCE or ASL224 recipes;
# ImageNet retains its iBOT-scaled LR 0.004 and separate feature/transform recipe.
# Usage: sbatch slurm/evaluation_classification.sh /absolute/path/launch_config.yaml

#SBATCH --job-name=eval-crisp-classification
#SBATCH --partition=2080ti,3090,3090_risk,a100,rtx5000,rtx8000,rtx_a6000_risk,l40s_risk,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=output/evaluation_%j.out
#SBATCH --error=output/evaluation_%j.err
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG_PATH="${1:-config/evaluation.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Evaluation configuration not found: ${CONFIG_PATH}" >&2
    exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
requeue_before_timeout() {
    trap '' USR1
    echo "Approaching the Slurm time limit; requeueing ${SLURM_JOB_ID} to resume evaluation checkpoints."
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_before_timeout USR1
echo "Node: $(hostname)"
echo "Config: ${CONFIG_PATH}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
srun --ntasks=1 --cpu-bind=cores \
    ./.conda-env/bin/python -u evaluation.py "${CONFIG_PATH}" &
wait "$!"
