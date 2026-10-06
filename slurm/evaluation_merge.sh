#!/usr/bin/env bash
#SBATCH --job-name=eval-summary
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
./.conda-env/bin/python -u -m evaluation.merge_results "$1" --final
