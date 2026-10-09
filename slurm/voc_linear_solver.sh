#!/usr/bin/env bash
#SBATCH --job-name=voc-linear-convergence
#SBATCH --partition=debug,2080ti,rtx5000,rtx8000,3090
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=output/voc_linear_solver_%j.out
#SBATCH --error=output/voc_linear_solver_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
SWEEP_CACHE="$1"
SWEEP_OUTPUT="$2"
shift 2
srun ./.conda-env/bin/python evaluation/voc_linear_solver.py --cache "$SWEEP_CACHE" --output "$SWEEP_OUTPUT" "$@"
