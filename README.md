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
In `evaluation.yaml`, set `datasets_root` to the prepared-data path, select your `checkpoint`,
and keep `output_dir: null` for separate results per launch.
