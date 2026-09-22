#!/usr/bin/env bash
# Submit all twelve ablations from the repository root with:
#   sbatch slurm/ablation.sh

#SBATCH --job-name=ibot-ablation
#SBATCH --array=0-13
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
# aisurrey14 currently cannot initialize the CUDA driver for this account.
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
    region_min_area_0p10 region_min_area_0p20 region_min_area_0p30 region_min_area_0p50
    lambda3_0p20 lambda3_0p50 lambda3_1p0 lambda3_2p0
    region_patch_threshold_0p2 region_patch_threshold_0p5 region_patch_threshold_0p8
    region_patch_threshold_weighted lambda3_0p10 region_min_area_1p0
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#configs[@]} )); then
    echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

repo_root="${SLURM_SUBMIT_DIR}"
cd "$repo_root"
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
