#!/usr/bin/env bash
# Run the configured full evaluation on one node with four GPUs.
#
# Usage from the repository root:
#   sbatch slurm/evaluation.sh config/evaluation.yaml
#
# The YAML controls the checkpoint, dataset paths, output directory, and
# selected evaluations.  Each evaluation worker discovers and uses all four
# GPUs allocated by Slurm.

#SBATCH --job-name=ibot-evaluation
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --nice=1000
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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072

requeue_before_timeout() {
    trap '' USR1
    echo "Approaching the Slurm time limit; requeueing ${SLURM_JOB_ID} to resume evaluation checkpoints."
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_before_timeout USR1

echo "Node:       $(hostname)"
echo "Config:     ${CONFIG_PATH}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-managed by Slurm}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# evaluation.py launches one distributed worker group per selected task using
# the GPUs visible in this allocation; do not wrap this command in torchrun.
srun --ntasks=1 --cpu-bind=cores \
    ./.conda-env/bin/python -u evaluation.py "${CONFIG_PATH}" &
wait "$!"
