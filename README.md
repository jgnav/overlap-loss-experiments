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

In `config/train.yaml`, set `data_path` to `<prepared-data-path>/imagenet/train` and
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

`ibot_plus_plus: false` keeps the original iBOT patch objective exactly: only
masked patches are distilled. Set it to `true` to add the same-view visible
patches to that objective, as in TIPSv2 iBOT++; the teacher remains detached and
centered, and the student uses the existing temperature-scaled log-softmax.
The masked and visible terms are normalized separately and summed, preserving
the original masked-patch signal. The result also reports `patch_masked`,
`patch_visible`, and `patch_all` metrics.

`koleo_regularizer: false` leaves the original objective unchanged. With
`true`, the [DINOv2 KoLeo regularizer](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py)
L2-normalizes each student's pre-head CLS feature, finds its nearest other
feature in the local batch, and minimizes the negative log distance. The two
global crops are processed separately so two views of the same image cannot
be nearest neighbors. The training objective adds 0.1 times the **sum** of
their KoLeo losses, matching [DINOv2's training integration](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py).
The option requires two global crops and at least two images per GPU; KoLeo
runs in float32 even under FP16/BF16 autocast.

The additional region-composition branch uses **raw, uncentered patch-head
logits** from the same two global views (including the existing masked student
forward). Crop geometry and horizontal flips map their shared region to each
patch grid. Patches participate when their covered fraction is at least
`region_patch_threshold` (default `0.5`); every selected patch has equal weight.
Alternatively, set `region_patch_threshold: weighted` to include every patch
with positive overlap and weight its representation by the fraction of its
area inside the overlap. The region mean is `sum(weight * representation) /
sum(weight)`: full coverage has weight 1, half coverage has weight 0.5, and
zero coverage contributes nothing. This applies to both student and teacher
pooling in all normalization modes; Sinkhorn assignments still balance the
included patches before their weighted region pooling. The `region_min_area`
filter remains active. Numeric thresholds preserve the existing binary rule.
`region_aggregation: mean` preserves the current mean pooling. For the
available moment/distribution ablations and their fixed settings, see
[region aggregation](docs/region_aggregation.md). Patch thresholding or area
weighting is applied before aggregation.

`region_normalization` selects the overlap representation:

- `centering`: reuse the ordinary iBOT teacher patch targets after subtracting
  `center2` and applying softmax at `teacher_patch_temp`. Student patches use
  the ordinary `student_temp` softmax. The branch averages selected patch
  distributions and applies symmetric cross-entropy. No centered targets are
  recomputed, and `region_temp` is unused.
- `softmax` (default): per-patch softmax at `region_temp`, then region pooling
  with the configured patch weights and symmetric cross-entropy.
- `raw_logits`: L2-normalize each raw patch vector, average selected vectors,
  then use symmetric cosine distance between the means. These vectors can be
  signed, so probability cross-entropy does not apply. `region_temp` is unused.
- `raw_logits_deep`: at blocks 3, 6, 9, and 12, average selected
  patch features after parameter-free LayerNorm at intermediate depths (the
  final block uses the ViT output norm). L2-normalize each regional mean,
  match student against the opposite teacher crop with cosine distance, and
  average the four layer losses. No projection head is used by this branch.
- `softmax_deep`: use the same blocks and shared binary overlap masks, project
  intermediate patches through the existing iBOT patch head, apply per-patch
  softmax at `region_temp`, average distributions within each region, and
  cross-distill opposite crops. Four layer losses are averaged. The patch
  head is shared across depths; no extra trainable heads are introduced.
  The forward pass pools logits in patch chunks, but student autograd still
  retains activations needed for backward; no activation checkpointing is used.
- `sinkhorn`: independently balance teacher and student selected patch logits
  using three Sinkhorn iterations at `region_temp`. Each side pools selected
  patches from both views and all ranks into one assignment problem, then
  averages assignments per region and uses symmetric cross-entropy. Gradients
  flow through student Sinkhorn, including its distributed normalization.

Deep modes require a numeric patch threshold and `region_aggregation: mean`.
The student uses masked global crops and the teacher uses unmasked crops.

The teacher is detached in every mode. Only patches from valid pairs passing
both geometry filters participate in Sinkhorn. DINO and iBOT always retain
their centered teacher softmax targets.

Set `register: 4` in the training YAML to insert four DINOv2-style learnable
memory tokens between CLS and the spatial patches. Registers participate in
self-attention but are excluded from the CLS/patch heads and every spatial
loss. `register: 0` preserves the original iBOT token sequence.

`lambda3` weights this additional loss;
`lambda3: 0.0` disables the branch; with KoLeo and iBOT++ off, this is the unchanged iBOT baseline.
`region_min_area` keeps the existing minimum intersection-area filter.
Pairs with no selected patches in either view are skipped. Use a new
continuation run for this changed objective: `resume_checkpoint: null`, with
`initial_checkpoint` pointing to the desired full teacher/student checkpoint.

In `config/evaluation.yaml`, set `datasets_root` to the prepared-data path, select your `checkpoint`,
and keep `output_dir: null` for separate results per launch.

For a paper-style A+B composition test from a COCO image ID and datasets root,
configure the checkpoint, image ID, and datasets root in `composition_visualization.py`.
By default it selects independent pure reference regions from COCO; explicit
`REFERENCE_A/B` lists remain available as an override. It forwards
a real crop containing both objects and fits the full-dimensional representation
with independently estimated A/B fingerprints. Then run:

```bash
python composition_visualization.py
```

See [the composition figure protocol](docs/composition_visualization.md) for
mask colors, fingerprint concentration, full-dimensional decomposition, the
separately forwarded mixed crop, and output details.
