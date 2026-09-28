#!/usr/bin/env bash
# Submit from the repository root with a walltime allowed by the target cluster:
#   sbatch --time=<cluster-allowed-walltime> slurm/slurm_long_training.sh
# Override --time, --cpus-per-task, or --mem at submission if needed.

#SBATCH --job-name=ibot-long
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
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
if [[ ! -f "$config_path" ]]; then
    echo "Training configuration not found: $config_path" >&2
    exit 2
fi

export IBOT_RUN_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

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

"$torchrun_bin" \
    --standalone \
    --nproc_per_node=8 \
    train.py "$config_path"
