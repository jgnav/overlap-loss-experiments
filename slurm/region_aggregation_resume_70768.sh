#!/usr/bin/env bash
# Resume one affected task from array 70768 in its original output and W&B run.
# Usage: sbatch slurm/region_aggregation_resume_70768.sh <1|3|4|5>

#SBATCH --job-name=ibot-region-aggregation-resume
#SBATCH --partition=a100,rtx_pro6000_risk
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/abaltion/resume_%j.out
#SBATCH --error=logs/abaltion/resume_%j.err

set -euo pipefail

task="${1:?Specify original task ID: 1, 3, 4, or 5}"
case "$task" in
    1) name=region_aggregation_mean_scalar_variance ;;
    3) name=region_aggregation_mean_covariance ;;
    4) name=region_aggregation_mean_projected_variance ;;
    5) name=region_aggregation_mean_projected_covariance ;;
    *) echo "Only array 70768 tasks 1, 3, 4, and 5 are supported" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR}"
source slurm/training_runtime.sh
configure_training_compatibility

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export IBOT_RUN_ID="70768_${task}"
unset IBOT_PRECISION_OVERRIDE IBOT_BATCH_SIZE_PER_GPU_OVERRIDE IBOT_GPU_COUNT_OVERRIDE

run_dir="output/ablation/${IBOT_RUN_ID}"
config_path="config/ablations/${name}.yaml"
runtime_config="${run_dir}/runtime_config.yaml"
[[ -s "${run_dir}/checkpoint.pth" ]] || { echo "Missing resume checkpoint in ${run_dir}" >&2; exit 2; }
compgen -G "${run_dir}/wandb/run-*" >/dev/null || { echo "Missing original W&B run ID in ${run_dir}" >&2; exit 2; }

prepare_training_config ./.conda-env/bin/python "$config_path" "$runtime_config" "$run_dir"
./.conda-env/bin/python - "$runtime_config" "$run_dir" <<'PY'
import pathlib
import sys
import yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
run_dir = pathlib.Path(sys.argv[2])
assert pathlib.Path(config["resume_checkpoint"]) == run_dir / "checkpoint.pth"
assert config["reset_optimizer"] is False
assert config["wandb_run_id"] and config["wandb_resume"] == "allow"
assert config["online_probe_batch_size"] == 128
PY

echo "Resuming ${IBOT_RUN_ID} from ${run_dir}/checkpoint.pth on $(hostname)"
exec ./.conda-env/bin/torchrun --standalone --nproc_per_node=4 train.py "$runtime_config"
