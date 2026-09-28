#!/usr/bin/env bash
# Resume array 76207 tasks 0-5 in their original output and W&B runs.
# Submit with: sbatch slurm/slurm_ablation_resume_76207.sh

#SBATCH --job-name=ablation-resume
#SBATCH --array=0-5
#SBATCH --partition=3090_risk,a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --output=logs/ablation_resume_%A_%a.out
#SBATCH --error=logs/ablation_resume_%A_%a.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"
cd "$SLURM_SUBMIT_DIR"

configs=(
    ibot_plus_plus_true
    koleo_regularizer_true
    lambda3_0p1
    lambda3_0p2
    lambda3_0p5
    lambda3_1p0
)
task="$SLURM_ARRAY_TASK_ID"
if (( task < 0 || task >= ${#configs[@]} )); then
    echo "Unsupported task ID: $task" >&2
    exit 2
fi

export IBOT_RUN_ID="76207_${task}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
run_dir="output/ablation/${IBOT_RUN_ID}"
base_config="config/ablations/${configs[$task]}.yaml"
runtime_config="${run_dir}/resume_config.yaml"

[[ -s "${run_dir}/checkpoint.pth" ]] || { echo "Missing checkpoint: ${run_dir}/checkpoint.pth" >&2; exit 2; }
[[ -f "$base_config" ]] || { echo "Missing config: $base_config" >&2; exit 2; }

./.conda-env/bin/python - "$base_config" "$run_dir" "$runtime_config" <<'PY'
import os
import pathlib
import sys
import tempfile
import yaml

base, run_dir, output = map(pathlib.Path, sys.argv[1:])
config = yaml.safe_load(base.read_text(encoding="utf-8"))
runs = sorted(path for path in (run_dir / "wandb").glob("run-*") if path.is_dir())
if not runs:
    raise SystemExit(f"Missing original W&B run in {run_dir / 'wandb'}")
run_id = runs[-1].name.rsplit("-", 1)[-1]
config["resume_checkpoint"] = str((run_dir / "checkpoint.pth").resolve())
config["reset_optimizer"] = False
config["wandb_run_id"] = run_id
config["wandb_resume"] = "allow"
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                 prefix=f".{output.name}.", delete=False) as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
    temporary = pathlib.Path(handle.name)
os.replace(temporary, output)
print(f"Resuming {run_dir} from {config['resume_checkpoint']} with W&B run {run_id}")
PY

exec ./.conda-env/bin/torchrun --standalone --nproc_per_node=4 \
    train.py "$runtime_config"
