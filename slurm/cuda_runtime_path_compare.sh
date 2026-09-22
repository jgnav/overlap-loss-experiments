#!/usr/bin/env bash
#SBATCH --job-name=ibot-cuda-path
#SBATCH --partition=3090_risk
#SBATCH --nodelist=aisurrey14
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/abaltion/cuda_path_%j.out
#SBATCH --error=logs/abaltion/cuda_path_%j.err

set -u
cd "${SLURM_SUBMIT_DIR}"
source slurm/training_runtime.sh
configure_training_compatibility

echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1

for label in clean runtime_libs; do
    echo "=== ${label} ==="
    if [[ "$label" == "clean" ]]; then
        env -u LD_LIBRARY_PATH ./.conda-env/bin/python - <<'PY'
import ctypes, torch
c = ctypes.CDLL("libcuda.so.1")
e = c.cuInit(0)
print("cuInit:", e)
print("torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
PY
    else
        env LD_LIBRARY_PATH="$PWD/.runtime-libs" ./.conda-env/bin/python - <<'PY'
import ctypes, torch
c = ctypes.CDLL("libcuda.so.1")
e = c.cuInit(0)
print("cuInit:", e)
print("torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
PY
    fi
    echo "status=$?"
done
