#!/usr/bin/env bash
# Re-evaluate only the saved epoch-50 teacher from register ablation 83657_22.
# The evaluator reuses the already completed VOC linear result in epoch0050/.
#SBATCH --job-name=register-e50-probes
#SBATCH --partition=a100,rtx_pro6000_risk
#SBATCH --exclude=aisurrey37
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/register_epoch50_%j.out
#SBATCH --error=logs/register_epoch50_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

probe_root=output/ablation/83657_22/online_probes
checkpoint="$probe_root/checkpoints/teacher_epoch0050.pth"
[[ -s "$checkpoint" ]] || { echo "Missing epoch-50 teacher snapshot: $checkpoint" >&2; exit 2; }

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "Evaluating $checkpoint on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --ntasks=1 --cpu-bind=cores ./.conda-env/bin/python -u -m evaluation.online_probes \
    --checkpoint "$checkpoint" \
    --datasets-root /mnt/fast/nobackup/scratch4weeks/jg02228/datasets \
    --output "$probe_root/epoch0050.json" \
    --arch vit_small --seed 0 --batch-size 128 --num-workers 0 \
    --epoch 50 --frequency 5
