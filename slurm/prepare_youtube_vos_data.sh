#!/usr/bin/env bash
# Fetch the public YouTube-VOS 2019 validation archive from its official Drive
# release. Offline scoring additionally requires complete validation masks.

#SBATCH --job-name=prepare-ytvos-eval
#SBATCH --partition=2080ti,3090,a100,rtx8000,rtx5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=output/prepare_ytvos_eval_%j.out
#SBATCH --error=output/prepare_ytvos_eval_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

datasets_root=/mnt/fast/nobackup/scratch4weeks/jg02228/datasets
downloads="$datasets_root/downloads/offline_evaluation"
archive="$downloads/youtube_vos_2019_valid.tar"
destination="$datasets_root/youtube_vos_2019"
mkdir -p "$downloads" "$destination"

echo "Downloading official YouTube-VOS 2019 validation archive"
./.conda-env/bin/gdown --continue --retries 5 \
    --output "$archive" 1bw8KcpzfrT08HYbuROZmY0bp4TkYl4_g
test "$(stat -c '%s' "$archive")" = 1296824320
tar -tf "$archive" > "$downloads/youtube_vos_2019_valid_contents.txt"
tar -xf "$archive" -C "$destination" --no-same-owner
echo "YouTube-VOS validation archive extracted to $destination"
