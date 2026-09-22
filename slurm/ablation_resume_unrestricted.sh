#!/usr/bin/env bash
# Resume one ablation task without excluding any cluster node.
# Usage: sbatch slurm/ablation_resume_unrestricted.sh <task-id>

#SBATCH --job-name=ibot-ablation-resume-unrestricted
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/abaltion/resume_unrestricted_%j.out
#SBATCH --error=logs/abaltion/resume_unrestricted_%j.err

set -euo pipefail

task="${1:?Specify original task ID}"
cd "${SLURM_SUBMIT_DIR}"

# The SBATCH directives in ablation_resume.sh are comments when invoked by
# bash; its task mapping, checkpoint recovery, output directory, and W&B
# recovery behavior remain unchanged.
exec bash slurm/ablation_resume.sh "${task}"
