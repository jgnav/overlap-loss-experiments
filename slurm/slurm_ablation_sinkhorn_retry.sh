#!/usr/bin/env bash
# Retry the original Sinkhorn run, which failed before its first checkpoint.
# Restrict it to RTX Pro 6000 nodes and keep its output directory and W&B ID.
#SBATCH --job-name=ablation-sinkhorn-retry
#SBATCH --partition=rtx_pro6000_risk
#SBATCH --nodelist=aisurrey36,aisurrey38
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=50:00:00
#SBATCH --output=logs/ablation_83650_6.out
#SBATCH --error=logs/ablation_83650_6.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

export IBOT_RUN_ID=83650_6
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
[[ ! -e output/ablation/83650_6/checkpoint.pth ]] || {
    echo 'Sinkhorn now has a checkpoint; use the full-resume launcher.' >&2
    exit 2
}
exec ./.conda-env/bin/torchrun --standalone --nproc_per_node=4 train.py \
    output/ablation/83650_6/retry_config.yaml
