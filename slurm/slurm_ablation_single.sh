#!/bin/bash
# Submit one ablation from the repository root:
# sbatch --job-name=ablation-lambda0 slurm/slurm_ablation_single.sh config/ablations/lambda3_0p0.yaml

#SBATCH --job-name=ablation-single
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
: "${SLURM_JOB_ID:?Missing Slurm job ID}"

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch slurm/slurm_ablation_single.sh config/ablations/<name>.yaml" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"
config_path="$1"
if [[ ! -f "$config_path" ]]; then
    echo "Training configuration not found: $config_path" >&2
    exit 2
fi

export IBOT_RUN_ID="$SLURM_JOB_ID"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Config: $config_path"
echo "Run ID: $IBOT_RUN_ID"
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi

./.conda-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py "$config_path"
