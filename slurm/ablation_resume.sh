#!/usr/bin/env bash
# Resume a requeued ablation task while retaining its original output directory
# and W&B run. For example: sbatch slurm/ablation_resume.sh 2

#SBATCH --job-name=ibot-ablation-resume
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
# These nodes currently fail CUDA driver initialization before training starts.
#SBATCH --exclude=aisurrey14,aisurrey36
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/abaltion/resume_%j.out
#SBATCH --error=logs/abaltion/resume_%j.err

set -euo pipefail

task="${1:?Specify original task ID}"
case "$task" in
    0)
        original_run_id="65866_0"
        config_path="config/ablations/region_min_area_0p10.yaml"
        ;;
    1)
        original_run_id="67958_1"
        config_path="config/ablations/region_min_area_0p20.yaml"
        ;;
    2)
        original_run_id="65866_2"
        config_path="config/ablations/region_min_area_0p30.yaml"
        ;;
    3)
        original_run_id="65948_3"
        config_path="config/ablations/region_min_area_0p50.yaml"
        ;;
    5)
        original_run_id="65866_5"
        config_path="config/ablations/lambda3_0p50.yaml"
        ;;
    9)
        original_run_id="65866_9"
        config_path="config/ablations/region_patch_threshold_0p5.yaml"
        ;;
    10)
        original_run_id="65866_10"
        config_path="config/ablations/region_patch_threshold_0p8.yaml"
        ;;
    12)
        original_run_id="66553_12"
        config_path="config/ablations/lambda3_0p10.yaml"
        ;;
    13)
        original_run_id="68508_13"
        config_path="config/ablations/region_patch_threshold_1p0.yaml"
        ;;
    *)
        echo "Unknown task ID: $task (expected 1, 2, 3, 5, 9, 10, 12, or 13)" >&2
        exit 2
        ;;
esac

repo_root="${SLURM_SUBMIT_DIR}"
cd "$repo_root"
source slurm/training_runtime.sh
configure_training_compatibility

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
# Keep output/ablation/<original_run_id> rather than creating a new job-ID directory.
export IBOT_RUN_ID="$original_run_id"
unset IBOT_PRECISION_OVERRIDE IBOT_BATCH_SIZE_PER_GPU_OVERRIDE IBOT_GPU_COUNT_OVERRIDE

echo "Resuming original ablation task: $original_run_id"
echo "Config: $config_path"
echo "Output: output/ablation/$original_run_id"
echo "Node: $(hostname)"
run_dir="output/ablation/$original_run_id"
runtime_config="$run_dir/runtime_config.yaml"
prepare_training_config ./.conda-env/bin/python "$config_path" "$runtime_config" "$run_dir"

exec ./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py "$runtime_config"
