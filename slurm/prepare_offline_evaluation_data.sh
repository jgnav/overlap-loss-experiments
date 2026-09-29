#!/usr/bin/env bash
# Download and prepare the public NAVI, DAVIS 2017, and Visual Genome inputs
# needed by config/evaluation.yaml, including the local SPair-71k archive.

#SBATCH --job-name=prepare-offline-eval
#SBATCH --partition=2080ti,3090,a100,rtx8000,rtx5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=output/prepare_offline_eval_%j.out
#SBATCH --error=output/prepare_offline_eval_%j.err

set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

datasets_root=/mnt/fast/nobackup/scratch4weeks/jg02228/datasets
downloads="$datasets_root/downloads/offline_evaluation"
mkdir -p "$downloads" "$datasets_root/visual_genome/vg500_annotations"

echo "Preparing SPair-71k from the local archive"
if [[ ! -d "$datasets_root/SPair-71k/PairAnnotation" ]]; then
    tar -xzf "$datasets_root/SPair-71k.tar.gz" -C "$datasets_root" --no-same-owner
fi
test -d "$datasets_root/SPair-71k/JPEGImages"
test -d "$datasets_root/SPair-71k/PairAnnotation"

fetch_exact() {
    local url="$1" destination="$2" expected_bytes="$3"
    local actual_bytes=0
    if [[ -f "$destination" ]]; then
        actual_bytes=$(stat -c '%s' "$destination")
    fi
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
            --output "$destination" "$url"
    fi
    actual_bytes=$(stat -c '%s' "$destination")
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        echo "Unexpected download size for $destination: $actual_bytes != $expected_bytes" >&2
        exit 1
    fi
}

echo "Preparing DAVIS 2017 train/validation at 480p"
fetch_exact \
    https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip \
    "$downloads/DAVIS-2017-trainval-480p.zip" 832766765
unzip -tq "$downloads/DAVIS-2017-trainval-480p.zip"
unzip -nq "$downloads/DAVIS-2017-trainval-480p.zip" -d "$datasets_root"
test -f "$datasets_root/DAVIS/ImageSets/2017/val.txt"
test -d "$datasets_root/DAVIS/JPEGImages/480p"
test -d "$datasets_root/DAVIS/Annotations/480p"

echo "Preparing NAVI v1"
fetch_exact \
    https://storage.googleapis.com/gresearch/navi-dataset/navi_v1.tar.gz \
    "$downloads/navi_v1.tar.gz" 31098738677
tar -xzf "$downloads/navi_v1.tar.gz" -C "$datasets_root" --no-same-owner
test -d "$datasets_root/navi_v1"
./.conda-env/bin/python -m evaluation.prepare_navi "$datasets_root/navi_v1" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "Preparing Visual Genome images"
fetch_exact \
    https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip \
    "$downloads/VG_images.zip" 9731705982
fetch_exact \
    https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip \
    "$downloads/VG_images2.zip" 5471658058
unzip -tq "$downloads/VG_images.zip"
unzip -tq "$downloads/VG_images2.zip"
unzip -nq "$downloads/VG_images.zip" -d "$datasets_root/visual_genome"
unzip -nq "$downloads/VG_images2.zip" -d "$datasets_root/visual_genome"

echo "Preparing the public SSGRL VG500 split and labels"
annotations="$datasets_root/visual_genome/vg500_annotations"
for filename in train_list_500.txt test_list_500.txt vg_category_500_labels_index.json; do
    curl --fail --location --retry 5 --retry-delay 5 \
        --output "$annotations/$filename" \
        "https://raw.githubusercontent.com/HCPLab-SYSU/SSGRL/master/data/VG/$filename"
done
./.conda-env/bin/python -m evaluation.prepare_visual_genome_manifest \
    --datasets-root "$datasets_root" --annotations-dir "$annotations"

echo "Public offline evaluation inputs prepared"
