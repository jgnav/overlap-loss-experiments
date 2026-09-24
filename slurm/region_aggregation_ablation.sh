#!/usr/bin/env bash
# Submit the ten region aggregation ablations from the repository root with:
#   sbatch slurm/region_aggregation_ablation.sh

#SBATCH --job-name=ibot-region-aggregation
#SBATCH --array=0-9
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey14
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/abaltion/%A_%a.out
#SBATCH --error=logs/abaltion/%A_%a.err

set -euo pipefail

: "${SLURM_ARRAY_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?The Slurm array task ID is missing}"

configs=(
    region_aggregation_mean
    region_aggregation_mean_scalar_variance
    region_aggregation_mean_variance
    region_aggregation_mean_covariance
    region_aggregation_mean_projected_variance
    region_aggregation_mean_projected_covariance
    region_aggregation_swd
    region_aggregation_mean_centered_swd
    region_aggregation_mean_normalized_swd
    region_aggregation_hellinger
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#configs[@]} )); then
    echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR}"
source slurm/training_runtime.sh
configure_training_compatibility
name="${configs[$SLURM_ARRAY_TASK_ID]}"
config_path="config/ablations/${name}.yaml"
[[ -f "$config_path" ]] || { echo "Missing config: $config_path" >&2; exit 2; }

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export IBOT_RUN_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
unset IBOT_PRECISION_OVERRIDE IBOT_BATCH_SIZE_PER_GPU_OVERRIDE IBOT_GPU_COUNT_OVERRIDE

run_dir="output/ablation/$IBOT_RUN_ID"
mkdir -p "$run_dir"
cp "$config_path" "$run_dir/ablation.yaml"
runtime_config="$run_dir/runtime_config.yaml"
prepare_training_config ./.conda-env/bin/python "$config_path" "$runtime_config" "$run_dir"

echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Ablation: $name"
echo "Config: $config_path"
echo "Output: $run_dir"
echo "Node: $(hostname)"

exec ./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py "$runtime_config"
