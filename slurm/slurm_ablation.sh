#!/bin/bash
# Submit from the repository root in two stages:
#   first=$(sbatch --parsable --job-name=ablation-core --array=0-8 slurm/slurm_ablation.sh)
#   sbatch --job-name=ablation-rest --array=9-25 --dependency="after:${first}" slurm/slurm_ablation.sh
# The second array becomes eligible when all first-stage tasks have started.

#SBATCH --job-name=ablation
#SBATCH --array=0-25
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
: "${SLURM_ARRAY_JOB_ID:?This launcher requires a Slurm job array}"
: "${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"

cd "$SLURM_SUBMIT_DIR"

# Keep this list and --array in sync. One array task launches one training job.
configs=(
    lambda3_0p4
    region_aggregation_mean_scalar_variance
    region_aggregation_mean_variance
    region_aggregation_mean_covariance
    region_normalization_raw_logits
    region_normalization_sinkhorn
    region_normalization_softmax
    region_aggregation_hellinger
    lambda3_0p2
    lambda3_0p6
    lambda3_0p8
    lambda3_1p0
    ibot_plus_plus_true
    koleo_regularizer_true
    region_min_area_0p2
    region_min_area_0p3
    region_min_area_0p5
    region_patch_threshold_0p2
    region_patch_threshold_0p5
    region_patch_threshold_weighted
    register_4
    shared_head_false
    region_normalization_deep
    include_local_crops_true
    loss_modality_cross_image
    loss_modality_within_image
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
