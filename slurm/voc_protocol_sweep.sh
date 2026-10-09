#!/usr/bin/env bash
# Exploratory original-iBOT protocol comparisons, with an immutable feature cache.
#SBATCH --job-name=voc-ibot-protocol-sweep
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=output/voc_protocol_sweep_%j.out
#SBATCH --error=output/voc_protocol_sweep_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
SWEEP_TRANSFORM="${1:-square}"
SWEEP_OUTPUT="$2"
shift 2
srun ./.conda-env/bin/python evaluation/voc_protocol_sweep.py --transform "$SWEEP_TRANSFORM" --output "$SWEEP_OUTPUT" "$@"
