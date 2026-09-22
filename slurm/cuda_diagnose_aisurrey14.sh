#!/usr/bin/env bash
#SBATCH --job-name=ibot-cuda-diagnose
#SBATCH --partition=3090_risk
#SBATCH --nodelist=aisurrey14
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/abaltion/cuda_diagnose_%j.out
#SBATCH --error=logs/abaltion/cuda_diagnose_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/training_runtime.sh
configure_training_compatibility

echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

./.conda-env/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
