#!/bin/bash
# Submit all four view ablations from the repository root:
#   sbatch slurm/slurm_region_views.sh

#SBATCH --job-name=region-views
#SBATCH --array=0-3
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/region_views_%A_%a.out
#SBATCH --error=logs/region_views_%A_%a.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
: "${SLURM_ARRAY_JOB_ID:?This launcher requires a Slurm job array}"
: "${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"

cd "$SLURM_SUBMIT_DIR"

# Keep this list and --array in sync. One array task launches one training job.
configs=(
    region_views_global
    region_views_global_local
    region_views_local
    region_views_global_unmasked
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#configs[@]} )); then
    echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

config_path="config/ablations/${configs[$SLURM_ARRAY_TASK_ID]}.yaml"
if [[ ! -f "$config_path" ]]; then
    echo "Training configuration not found: $config_path" >&2
    exit 2
fi

# All ranks of this task share a unique output directory; separate tasks never
# overwrite one another's checkpoint or TensorBoard files.
export IBOT_RUN_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Array task: $SLURM_ARRAY_TASK_ID / ${#configs[@]}"
echo "Config: $config_path"
echo "Run ID: $IBOT_RUN_ID"
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi

./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py "$config_path"
