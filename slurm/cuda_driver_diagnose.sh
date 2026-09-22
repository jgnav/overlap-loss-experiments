#!/usr/bin/env bash
#SBATCH --job-name=ibot-cuda-driver
#SBATCH --partition=3090_risk
#SBATCH --nodelist=aisurrey14
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/abaltion/cuda_driver_%j.out
#SBATCH --error=logs/abaltion/cuda_driver_%j.err

set -u
cd "${SLURM_SUBMIT_DIR}"
source slurm/training_runtime.sh
configure_training_compatibility

echo "=== identity and environment ==="
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "=== driver version query ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
nvidia-smi

echo "=== device nodes ==="
ls -la /dev/nvidia* 2>&1 || true
for dev in /dev/nvidiactl /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidia-uvm; do
    echo "--- $dev ---"
    stat "$dev" 2>&1 || true
done

echo "=== loaded kernel module ==="
cat /proc/driver/nvidia/version 2>&1 || true
echo "=== libcuda target and linker cache ==="
readlink -f /lib64/libcuda.so.1 2>&1 || true
stat /lib64/libcuda.so.1 2>&1 || true
ldconfig -p 2>/dev/null | grep -E 'libcuda|libnvidia-ml' || true
echo "=== NVIDIA error state ==="
nvidia-smi -q -d ERROR 2>&1 || true

echo "=== possible stub/compat libraries ==="
find .conda-env -name 'libcuda.so*' -o -name 'libnvidia*.so*' 2>/dev/null | head -80
find /usr/local -path '*stubs/libcuda.so*' -o -path '*compat/libcuda.so*' 2>/dev/null | head -80

echo "=== libcuda resolution ==="
LD_DEBUG=libs ./.conda-env/bin/python -c 'import ctypes; ctypes.CDLL("libcuda.so.1")' 2>&1 | grep -E 'libcuda|libnvidia' || true

echo "=== direct cuInit ==="
./.conda-env/bin/python - <<'PY'
import ctypes

cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
err = cuda.cuInit(0)
name = ctypes.c_char_p()
desc = ctypes.c_char_p()
cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
cuda.cuGetErrorName(err, ctypes.byref(name))
cuda.cuGetErrorString(err, ctypes.byref(desc))
print("cuInit return code:", err)
print("error name:", name.value.decode() if name.value else None)
print("description:", desc.value.decode() if desc.value else None)
PY

echo "=== torch CUDA check ==="
./.conda-env/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
