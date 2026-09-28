#!/usr/bin/env bash
# Submit with an original run ID, keeping its original stdout/stderr paths:
# sbatch --output=logs/ablation_<run_id>.out --error=logs/ablation_<run_id>.err \
#   --open-mode=append slurm/slurm_ablation_full_resume.sh <run_id>

#SBATCH --job-name=ablation-full-resume
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=50:00:00
#SBATCH --output=logs/ablation_full_resume_%j.out
#SBATCH --error=logs/ablation_full_resume_%j.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
run_id="${1:?Pass the original ablation run ID}"
[[ "$run_id" =~ ^[0-9]+_[0-9]+$ ]] || { echo "Invalid run ID: $run_id" >&2; exit 2; }
cd "$SLURM_SUBMIT_DIR"

export IBOT_RUN_ID="$run_id"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
run_dir="output/ablation/$run_id"
runtime_config="$run_dir/resume_config.yaml"
[[ -s "$run_dir/checkpoint.pth" ]] || { echo "Missing checkpoint in $run_dir" >&2; exit 2; }
[[ -s "$runtime_config" ]] || { echo "Missing resume config: $runtime_config" >&2; exit 2; }

echo "Full resume $run_id on $(hostname), checkpoint: $run_dir/checkpoint.pth"
exec ./.conda-env/bin/torchrun --standalone --nproc_per_node=4 train.py "$runtime_config"
