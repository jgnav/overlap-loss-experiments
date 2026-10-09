#!/usr/bin/env bash
#SBATCH --job-name=semantic-region-retrieval
#SBATCH --partition=3090,rtx8000,a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=output/semantic_retrieval_%A_%a.out
#SBATCH --error=output/semantic_retrieval_%A_%a.err
set -euo pipefail
RUN_ROOT="${1:?Pass the run directory}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$RUN_ROOT/source"
CHECKPOINT="$SLURM_SUBMIT_DIR/checkpoints/ibot_vit_small.pth"
NAME=ibot_original
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then
  CHECKPOINT="$SLURM_SUBMIT_DIR/output/long_ibot_vit_small/85535_0/checkpoint_source1000_continuation0200.pth"
  NAME=region200
fi
srun "$SLURM_SUBMIT_DIR/.conda-env/bin/python" -u -m evaluation.semantic_region_retrieval \
  --checkpoint "$CHECKPOINT" --output "$RUN_ROOT/$NAME" \
  --voc /mnt/fast/nobackup/scratch4weeks/jg02228/datasets/pascal_voc/VOCdevkit/VOC2012
