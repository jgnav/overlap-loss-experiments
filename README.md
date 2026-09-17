# Setup

Run from the repository root. Requires Linux, Python 3.10, an NVIDIA driver,
and `wget`.

## 1. Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

## 2. Download datasets

Choose any folder for all downloaded archives:

```bash
DOWNLOAD_DIR="/path/to/downloads"
mkdir -p "$DOWNLOAD_DIR"

# ADE20K
wget -c -P "$DOWNLOAD_DIR" http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# PASCAL VOC2012 (no SBD needed)
wget -c -P "$DOWNLOAD_DIR" http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# COCO2017
wget -c -P "$DOWNLOAD_DIR" https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip
wget -c -P "$DOWNLOAD_DIR" https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
wget -c -P "$DOWNLOAD_DIR" https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
```

Download manually after logging in and accepting the dataset terms:

- [Kaggle ImageNet downloads](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data):
  `imagenet-object-localization-challenge.zip` (includes validation labels).
  Alternatively, [original ImageNet downloads](https://www.image-net.org/download.php):
  `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`,
  `ILSVRC2012_devkit_t12.tar.gz`.
- [Cityscapes downloads](https://www.cityscapes-dataset.com/downloads/):
  `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`.

## 3. Prepare datasets

Once all eight archives are downloaded (ten with the original ImageNet tars):

```bash
python prepare_data.py "$DOWNLOAD_DIR"
```

Data is prepared directly inside `$DOWNLOAD_DIR`, alongside the archives:
`imagenet/`, `ade20k/`, `pascal_voc/`, `cityscapes/`, `coco/`, and
`evaluation_manifests/`.
The script extracts archives, organizes ImageNet classes, creates VOC/COCO
manifests, and validates the data. Archives are retained. Preparation is
resumable: the script records completed phases in
`$DOWNLOAD_DIR/.prepare_data_state.json`, skips phases that finished in an
earlier run, and continues interrupted archive extraction from files already
written. A final validation pass still checks the complete dataset before it
reports success.
When resuming, unchanged archives retain their completed integrity check.
Existing ImageNet files with matching sizes are kept without opening their ZIP
members; missing or truncated files are extracted again. The archive index and
existing files still need scanning, which can take time for ImageNet. A missing
train/validation directory or either classification manifest makes its phase
eligible to run again. Only a successful final validation sets `validated: true`.
Kaggle ImageNet is detected automatically, including ZIPs containing a nested
ImageNet tar. Validation images are classified using `LOC_val_solution.csv`;
test images are skipped. No separate ImageNet devkit is needed for this format.

## 4. Download iBOT checkpoints

Full checkpoints from the [official iBOT repository](https://github.com/bytedance/ibot#pre-trained-models).
ViT-S/16 is used by the default training configuration.

```bash
mkdir -p checkpoints

# ViT-S/16
wget -c -O checkpoints/ibot_vit_small.pth \
  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint.pth

# Optional: ViT-B/16
wget -c -O checkpoints/ibot_vit_base.pth \
  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint.pth

# Optional: ViT-L/16
wget -c -O checkpoints/ibot_vit_large.pth \
  https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint.pth
```

In `train.yaml`, set `data_path` to `<prepared-data-path>/imagenet/train` and
`initial_checkpoint: checkpoints/ibot_vit_small.pth`.

`shared_head: true` uses one projection MLP and prototype layer for CLS and
patch tokens. With `shared_head: false`, both paths are independent. Loading
a shared-head iBOT checkpoint copies its complete head into both paths for
the student and teacher, and duplicates Adam moments into independent state.
The output dimensions must match the pretrained head to copy its weights.
Existing separate-head checkpoints retain their distinct prototype weights;
older checkpoints with a shared MLP initialize both MLPs from that saved MLP.
The global DINO and masked iBOT objectives always use teacher center + softmax,
with separate CLS and patch centers restored from checkpoints. Their student
outputs use the original temperature-scaled log-softmax. There is no teacher
normalization selector.

The additional region-composition branch uses **raw, uncentered patch-head
logits** from the same two global views (including the existing masked student
forward). Crop geometry and horizontal flips map their shared region to each
patch grid. Patches participate when their covered fraction is at least
`region_patch_threshold` (default `0.5`); every selected patch has equal weight.
One `region_normalization` setting applies to **both teacher and student**:

- `softmax` (default): per-patch softmax at `region_temp`, then an equal-weight
  mean and symmetric cross-entropy.
- `raw_logits`: L2-normalize each raw patch vector, average selected vectors,
  then use symmetric cosine distance between the means. These vectors can be
  signed, so probability cross-entropy does not apply. `region_temp` is unused.
- `sinkhorn`: independently balance teacher and student selected patch logits
  using three Sinkhorn iterations at `region_temp`. Each side pools selected
  patches from both views and all ranks into one assignment problem, then
  averages assignments per region and uses symmetric cross-entropy. Gradients
  flow through student Sinkhorn, including its distributed normalization.

The teacher is detached in every mode. Only patches from valid pairs passing
both geometry filters participate in Sinkhorn; no centering is used by any
region mode. DINO and iBOT always retain their centered teacher softmax targets.

`lambda3` weights this additional loss;
`lambda3: 0.0` disables the branch for the unchanged iBOT baseline.
`region_min_area` keeps the existing minimum intersection-area filter.
Pairs with no selected patches in either view are skipped. Use a new
continuation run for this changed objective: `resume_checkpoint: null`, with
`initial_checkpoint` pointing to the desired full teacher/student checkpoint.

In `evaluation.yaml`, set `datasets_root` to the prepared-data path, select your `checkpoint`,
and keep `output_dir: null` for separate results per launch.

For a paper-style A+B composition figure from a COCO image ID and datasets root, configure the checkpoint, image ID, datasets root, and independent reference regions
in `composition_visualization.py`
and run:

```bash
python composition_visualization.py
```

See [the composition figure protocol](docs/composition_visualization.md) for
mask colors, independent calibration, patch-count mixtures, and output details.
