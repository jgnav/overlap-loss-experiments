#!/bin/bash -l
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"

module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0
module load libffi/3.4.4-GCCcore-12.2.0

VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"

echo "Creating virtual environment in ${VENV_DIR}..."
python3 -m venv --clear "${VENV_DIR}"

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
python -m pip install -r requirements.txt

echo "Environment setup complete."
