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
The objective averages the cross-entropy uniformly over all patches. Separate
masked and visible means are diagnostics only. The result also reports `patch_masked`,
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

`include_local_crops: false` preserves the global-only regional objective.
Set it to `true` for the independent mean-pooling context ablation in
`config/ablations/include_local_crops_true.yaml` (the only changed setting).
The recipe still uses two 224-pixel global crops and ten 96-pixel local crops.
The teacher processes only globals; the student processes all crops, using
unmasked local forwards already required by DINO. For every global/local pair,
map their original-image intersection to their respective patch grids,
including horizontal flips. Pool only fully contained patches with an
unweighted arithmetic mean. A local pair is valid when its intersection has
positive area and both views contain at least one complete patch; the existing
`region_min_area` and `region_patch_threshold` filters still govern global/global.
Reuse the selected normalization and temperatures (ordinary iBOT teacher
targets for `centering`) and apply teacher-global to student-local CE.

Within **each image**, average valid global/local pair losses separately from
the symmetric global/global loss. Combine the two means with fixed weights
0.75 and 0.25; if only one group is valid, use its full loss. Exclude images
with neither group from the mean across images/GPUs. `lambda3` remains 0.4 in
the ablation YAML; DINO, ordinary iBOT and all other settings stay unchanged.
The option requires `region_aggregation: mean` and a probability normalization
(`centering`, `softmax` or `sinkhorn`); it is separate from the deep and moment
ablations. Sinkhorn balances unique teacher patches participating in valid
local intersections jointly across views/GPUs, then reuses their assignments
for every local pair. The existing global/global Sinkhorn problem is unchanged.
Logs include `region_global_global_loss`, `region_global_local_loss`,
`region_global_local_valid_ratio` and `region_global_local_pairs_per_image`.
Changing this flag requires a new continuation rather than exact resume.

`loss_modality: standard` preserves the existing objective. The two ordering
ablations change this selector and set `batch_size_per_gpu: 48` from `config/train.yaml`:

- `config/ablations/loss_modality_cross_image.yaml`: add ordering against teacher
  global/global overlap regions from other images in the distributed minibatch,
  with exactly one physical region per eligible image.
- `config/ablations/loss_modality_within_image.yaml`: add ordering against every
  other distinct eligible teacher region from the query image.

Both keep DINO, ordinary iBOT and the original regional loss unchanged, including
`lambda3: 0.4`, the source checkpoint, seed and 50-epoch continuation protocol.
Their objective is `L_DINO + L_iBOT + 0.4 L_region + 0.1 L_ordering`; the auxiliary
coefficient is fixed in code. Ordering requires mean aggregation, centering,
`region_patch_threshold: 1.0`, and two globals. Within-image ordering additionally
requires at least two locals. Teacher patch
probabilities reuse the ordinary centered iBOT targets; student patches use
their ordinary `student_temp` softmax, without teacher centering. The separate
`include_local_crops` flag keeps controlling the original regional CE branch;
within-image ordering automatically records all crop boxes and uses the existing
student local patch outputs, without changing that flag or adding backbone passes.
Cross-image ordering uses only global patch outputs and global geometry.

Cross-image ordering constructs only the intersection of the two global crops,
as in the standard region loss, and retains `region_min_area`. Each eligible
image supplies two cross-view queries and one detached teacher reference,
using the covering global with the most fully contained patches (ties choose
global 0). Both global grids must contain at least four selected patches.
Each query uses the one region from every other eligible image across all GPUs,
in a seeded shuffled order. Thus a minibatch with B eligible images provides
B - 1 references per query; with fewer than two references, skip ordering.
Local crops cannot change its regions, reference bank or ordering loss.

Within-image ordering enumerates global/global, global/local and local/local intersections in original
image coordinates, including flips. Deduplicate references by their exact
physical rectangle. A region must be fully covered by a global teacher crop
and have at least four fully contained patches in the teacher and both crop
views that define it. Small ordering regions use this patch-count criterion,
not `region_min_area`; the original regional loss retains its existing filter.
For each reference, use the covering global teacher with the most selected
patches (ties choose global 0). Global/global gives both cross-view directions;
global/local uses the paired teacher-global to student-local (keeping both
global teacher contexts when they cover the same region); local/local uses the best
covering global teacher to each participating student local. Duplicate physical
region/direction queries are counted once, with GG, then GL, then LL precedence.
Regions outside both global teachers are excluded; no local teacher pass is used.

For a within-image query with M other distinct regions, use all M references
within that image and skip if M < 2. It does not depend on external images and
can run with a single image in the minibatch. The two modalities now have
different query sets and reference counts by design. Teacher/student queries share the
same reference identities and input order. Teacher queries and references are
detached; cosine comparisons retain the complete prototype distribution.

The indexed bitonic sorter matches the NeCo-pinned `diffsort==0.2.0`, including
arbitrary sequence lengths, with teacher/student steepness 100. It uses that
released implementation's default Cauchy interpolation (the paper appendix
instead names logistic-phi). Cross-entropy sums over reference identities and
averages over ranks of the full M-by-M permutation matrices. This follows the
[released NeCo sorting and loss code](https://github.com/vpariza/NeCo/blob/main/src/neco.py)
and [diffsort implementation](https://github.com/Felix-Petersen/diffsort/tree/main/diffsort).
There is no new runtime dependency; the upstream MIT notice is preserved in
`third_party/diffsort_LICENSE`. Cosines use shared matrix multiplication for
cross-image banks and batched matrix multiplication for within-image banks,
as in NeCo. Gather reference identities only after computing scalar cosines;
never replicate the full prototype bank per query. Batch region pooling across
images, with 64 MiB softmax tiles and a single gradient allocation per crop view.
The tiled pooling recomputes softmax in backward and supports first-order
training gradients. Reuse duplicate student/teacher sorting queries; retain
detached teacher permutations and checkpoint only the student sorter. Group
matching grid sizes for geometry, and group query lengths without padding
reference identities/ranks. These are implementation optimizations: the
selected regions, reference order, full-dimensional predictions, sorter
parameters, coefficients and DDP reductions are unchanged. The CPU benchmark
`benchmarks/benchmark_region_ordering.py` measures forward/backward of the loss
without running a backbone or GPU; `--reference-source` can compare a saved
directory containing the previous ordering/sorting Python files.

Average valid queries separately for GG/GL/LL over all GPUs, then combine these
means with fixed weights 0.5/0.25/0.25, renormalizing for absent families.
Cross-image ordering has only GG queries, so it uses that family's full mean.
Logs include `region_ordering`, `region_ordering_raw`, query/reference counts,
mean references per query, and each family's loss and query count. Disabling
`lambda3` also disables ordering. A separate seeded sampler leaves augmentation
and mask randomness untouched; its step is restored on exact checkpoint resume,
alongside the existing teacher centers. Changing modalities requires a new
continuation, not an exact resume. Slurm array indices 24 and 25 run the new modes.

`region_normalization` selects the overlap representation:

- `centering`: reuse the ordinary iBOT teacher patch targets after subtracting
  `center2` and applying softmax at `teacher_patch_temp`. Student patches use
  the ordinary `student_temp` softmax. The branch averages selected patch
  distributions and applies symmetric cross-entropy. No centered targets are
  recomputed, and `region_temp` is unused.
- `softmax`: uncentered teacher per-patch softmax at `region_temp` and ordinary
  student softmax at `student_temp`, then region pooling with the configured
  patch weights and symmetric cross-entropy. This holds the student temperature
  fixed when comparing with centered teacher targets.
- `raw_logits`: L2-normalize each raw patch vector, average selected vectors,
  then use symmetric cosine distance between the means. These vectors can be
  signed, so probability cross-entropy does not apply. `region_temp` is unused.
- `sinkhorn`: balance only the detached teacher logits of selected overlap
  patches using three Sinkhorn iterations at `region_temp`. Both views and
  all ranks share one teacher assignment problem. Student patches use the
  ordinary `student_temp` softmax. Pool each side with the configured patch
  weights, then apply symmetric cross-entropy. Unselected patches do not enter
  Sinkhorn. This restores the earlier teacher-only normalization; checkpoints
  from the student-Sinkhorn objective cannot be resumed under this loss.
- `deep`: apply the centered region objective at four evenly spaced backbone
  depths (3/6/9/12 for ViT-S/B, 6/12/18/24 for ViT-L). Each intermediate depth
  has an independent copy of the complete iBOT patch projection/prototype head
  and its own teacher center. The final depth uses the ordinary iBOT patch head
  and center, reusing its teacher softmax targets. Every depth applies per-patch
  softmax before aggregation: centered teacher targets at `teacher_patch_temp`
  and student probabilities at `student_temp`; `region_temp` is unused.
  Apply the configured patch
  selection/weighting and aggregation at each depth, then average the four
  region losses before multiplying by `lambda3`. Only global crops use these
  heads. The CLS and ordinary iBOT losses still use the final-layer outputs.

The `deep` depth schedule follows the intermediate outputs in
[V-JEPA 2.1](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py).
Our projection heads and centered region cross-entropy use this repository's
iBOT formulation. Start with
`config/ablations/region_normalization_deep.yaml`, which changes only the
normalization selector from `config/train.yaml`. A new continuation copies
each intermediate head and center from the pretrained patch head and `center2`;
exact resume restores their independently trained weights, EMA teacher heads,
centers and optimizer state. Per-depth raw losses are logged as
`region_depth_<block>`. The former `raw_logits_deep` and `softmax_deep` selectors
are not accepted.

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
