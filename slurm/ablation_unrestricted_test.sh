#!/usr/bin/env bash
# Launch task 1 with the same ablation logic but without excluding any node.

#SBATCH --job-name=ibot-ablation-unrestricted
#SBATCH --array=1
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/abaltion/unrestricted_%A_%a.out
#SBATCH --error=logs/abaltion/unrestricted_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# SBATCH directives in the sourced launcher are comments at runtime, while its
# task mapping and isolated output/W&B setup remain identical.
exec bash slurm/ablation.sh
