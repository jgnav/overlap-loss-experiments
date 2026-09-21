#!/usr/bin/env bash
# Resume a requeued ablation task while retaining its original output directory
# and W&B run. For example: sbatch slurm/ablation_resume.sh 2

#SBATCH --job-name=ibot-ablation-resume
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
# aisurrey14 currently exposes allocated GPUs but cannot initialize CUDA.
#SBATCH --exclude=aisurrey14
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/abaltion/resume_%j.out
#SBATCH --error=logs/abaltion/resume_%j.err

set -euo pipefail

task="${1:?Specify original task ID: 1 or 2}"
case "$task" in
    1)
        original_run_id="65866_1"
        config_path="config/ablations/region_min_area_0p20_resume.yaml"
        ;;
    2)
        original_run_id="65866_2"
        config_path="config/ablations/region_min_area_0p30_resume.yaml"
        ;;
    5)
        original_run_id="65866_5"
        config_path="config/ablations/lambda3_0p50_resume.yaml"
        ;;
    *)
        echo "Unknown task ID: $task (expected 1, 2, or 5)" >&2
        exit 2
        ;;
esac

repo_root="${SLURM_SUBMIT_DIR}"
cd "$repo_root"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
# Keep output/ablation/<original_run_id> rather than creating a new job-ID directory.
export IBOT_RUN_ID="$original_run_id"
unset IBOT_PRECISION_OVERRIDE IBOT_BATCH_SIZE_PER_GPU_OVERRIDE IBOT_GPU_COUNT_OVERRIDE

echo "Resuming original ablation task: $original_run_id"
echo "Config: $config_path"
echo "Output: output/ablation/$original_run_id"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec ./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py "$config_path"
