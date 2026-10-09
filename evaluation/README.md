# Evaluation protocols

Each evaluation has one implementation and one fixed recipe. There are no
protocol mode switches. Settings follow CRISP first, then applicable CAPI details, then iBOT.
ImageNet and multilabel linear classification use CRISP resolution, explicit learning
rate, GPU/batch counts and epoch budgets with documented iBOT/local details. Segmentation uses CAPI classifiers
and CRISP resolution. Other tasks use documented CRISP/iBOT or task-specific
components. Missing author details remain explicit assumptions.

Training-time monitoring uses these same evaluators for PASCAL VOC k-NN and
linear segmentation and ImageNet 10% CLS k-NN classification, including their
full parameter searches. See [`docs/online_probes.md`](../docs/online_probes.md)
for the enable switch and epoch interval.

## Selected segmentation protocol

Segmentation calls the vendored official CAPI evaluator, pinned to revision
`98b4fa17ee8eec8810c17022df9a27a44845368b`. Its classifiers, hyperparameter
selection, refitting and scoring are copied from upstream; local adapters
handle dataset paths, final normalized patch features, logging and JSON output.
Equal-scoring parameter choices retain the original grid order across GPU counts.
See [vendor provenance and integration changes](vendor/capi/README.md).

**CRISP's resolution adjustment** is retained: 256 x 256 for patch size 16
and 224 x 224 for patch size 14. Both produce a 16 x 16 grid (256 tokens).
This intentionally differs from upstream CAPI's fixed 224-pixel segmentation
input. Labels are patchified using the actual grid and patch
size. Final normalized teacher patch features follow CAPI's released iBOT loader.
Registers are excluded from spatial features. Compatible iBOT-style ViT-S/B/L
checkpoints are supported; this does not promise compatibility with arbitrary
architectures. Classification also uses 224 x 224.

The released CAPI `default_eval_config.yaml` uses bfloat16 nearest-neighbor
search for ADE20K; that override is applied. VOC/Cityscapes use the evaluator's
float32 default. Backbone feature extraction and cuML logistic regression remain
float32; TF32 is enabled as in CAPI's runtime setup.

ADE20K, PASCAL VOC 2012 and Cityscapes use frozen final-block patch features,
train-only StandardScaler, a seeded 10% training holdout for selecting probe
hyperparameters, and refitting on train plus holdout before evaluating the
official validation set. k-NN tests k in {1, 3, 10, 30} with cosine/L2 distance;
linear segmentation uses CAPI's cuML logistic-regression sweep. The paper-style
mIoU percentage is `metrics.miou_percent`.

VOC segmentation now uses **only the original VOC2012 splits**, as explicitly
requested: `ImageSets/Segmentation/train.txt` (1,464 images) and `val.txt`
(1,449 images), with original `SegmentationClass` PNG masks and official file
order. SBD is not discovered or loaded, even when installed; `trainaug` is
rejected. Duplicate IDs and train/val overlap fail validation. The internal
seeded holdout uses 146 training images, leaving 1,318 for probe selection;
the final probe is refitted on all 1,464 images. Official validation is never
used to fit or select probes.

The selected CAPI split is **train**, not **trainaug**. The released loader
supports both; neither CRISP's quoted passage nor CAPI Appendix H.2 establishes
which split the authors selected. Exact published-score reproduction is therefore
not established. The released CAPI `trainaug` loader also uses SBD; its concatenation
would produce 12,819 entries, 1,134 repetitions, and 1,103 official validation
images in training on our data; it also warns of non-reproduction of its paper.
Removing SBD is the selected experimental choice, not a claim about the
authors' unpublished image lists. Results record construction ID
`voc2012_original_segmentation_v1`, mask policy, zero training/validation
overlap, and ordered image/mask hashes under `dataset_manifests`.

Re-evaluate official iBOT, the continued-iBOT control, and the overlap model
under this same VOC-only protocol. Previous 100/200-epoch results used the
clean 10,582-image VOC+SBD split at 224 resolution; those are a different
protocol, not results of the later contaminated concatenation. Use a new
output directory to preserve provenance. Backbone retraining is not required.

### Other dataset split checks

ADE20K uses official `training` (20,210 images) and `validation` (2,000),
150 classes, ignoring labels 0 and 255. Cityscapes uses `leftImg8bit` with
`gtFine` train (2,975) and val (500), maps the 19 evaluation classes to train
IDs, and ignores other labels (255); coarse annotations and test images are
not used. Both retain the CAPI holdout/probes and CRISP's 256-token resolution adjustment. ImageNet classification uses official train/val with matching
1,000-class vocabularies; k-NN uses 1%, 10%, and 100% training banks and
linear probing uses the full official training split and reports on the full official
validation set. VOC multilabel classification remains
separate from segmentation: official VOC2012 `ImageSets/Main` train/val.
COCO remains the explicitly selected 2017 train/val baseline. Exact CRISP
classification split/recipe equivalence is **not established**; see below.

Changes to resolution, dataset list construction, seed, source code, or manifest
files invalidate old results. The pinned evaluator extracts fresh features for
each probe; legacy feature caches are not used. This can increase runtime
compared with reusing one feature bank across k-NN and linear tasks.
Source hashes cover the local evaluation and backbone Python files. They do not
hash every image's pixel content: use a new output directory if dataset files
are edited in place. Retain result JSON files with the evaluated checkpoints.

## Classification

### Linear classification: fixed multilabel benchmark protocol

`imagenet_linear` and the VOC/COCO/VG linear probes share the frozen linear
classifier in `evaluation/utils/classification.py`. They implement the settings
listed in CRISP Tables 3/4 and Appendix A.2, with selectable documented recipes.
The standalone comparison configs select the iBOT-style recipe described below.
The following whole-image conventions describe the retained `bce` recipe:

- **Exactly four GPUs**, **256 images per GPU** (global batch **1,024**).
  Any other GPU count is rejected before loading datasets or training probes.
- **224 x 224** inputs; the encoder remains frozen and only a single linear
  classifier is trained.
- Multilabel protocol version **5** uses **effective learning rate 0.001**,
  without automatic batch scaling. This is the user's agreed explicit
  interpretation of CRISP's incomplete LR description. ImageNet retains its
  separate iBOT convention: base 0.001 scaled by global batch / 256 = 0.004.
- Multilabel features concatenate the **final-block CLS token and mean spatial
  patch tokens**, both after final backbone LayerNorm and before projection.
  ViT-S features have 768 dimensions. There is no L2 normalization, softmax or
  StandardScaler. This pooling is a documented local choice; the supplied CRISP
  PDF does not specify the feature pooling.
- Multilabel training resizes the **entire image to 224 x 224 with bicubic
  interpolation**, followed by horizontal flip and ImageNet normalization.
  Evaluation uses the same whole-image resize without augmentation. This is an
  explicit completion choice, not a verified CRISP transform specification.
- **500 epochs** for VOC Full and VOC 1/2/5-shot; **200 epochs** for COCO,
  VG and ImageNet linear.
- Full-data tasks use their full training split and report on the full evaluation
  split. ImageNet uses every official training image and all 50,000 official
  validation images. No internal holdout or attentive classifier is used.
- VOC low-shot tasks randomly draw 1, 2 or 5 positive images per class from the
  training split using fixed seed 0, then report mAP on the full VOC validation
  set. Draws are nested across shot counts; the deduplicated union retains all
  labels of selected images. The authors' seed and multilabel overlap handling
  are not published.

| Evaluation | Training data | Epochs | Metric |
| --- | --- | --- | --- |
| `imagenet_linear` | Full official ImageNet-1K train, 1,000 classes | 200 | top-1/top-5 |
| `pascal_voc_multilabel` | VOC2012 classification train, 20 classes | 500 | mAP |
| `pascal_voc_1shot`, `pascal_voc_2shot`, `pascal_voc_5shot` | Random positive images per class from VOC train | 500 | mAP |
| `coco_multilabel` | COCO2017 train, 80 classes | 200 | mAP |
| `visual_genome_multilabel` | SSGRL VG500 train, 500 classes | 200 | mAP |

VOC uses official **VOC2010+ all-points interpolated precision-envelope AP**,
averaged equally over its 20 classes. Native difficult-only labels are ignored
per class in training and evaluation. COCO and VG use non-interpolated image-level
macro AP over their fixed vocabularies, not detection AP.

The standalone multilabel comparison configurations now select
`multilabel_recipe: ibot` (protocol version **8**). This restores the
iBOT-style adaptation used for the earlier Region200 VOC results
**49.59 / 64.39 / 74.28 / 89.31**: concatenate the last four normalized CLS
tokens for ViT-S (1,536 dimensions); train with RandomResizedCrop(224) and
horizontal flip; validate with Resize(256) and CenterCrop(224). Image-level
labels remain unchanged after cropping. SGD uses momentum **0.9**, zero weight
decay, cosine decay without warmup, and base LR **0.001** scaled by batch/256
to **0.004** at 4 GPUs x 256. Masked BCE, non-interpolated macro AP for all
three datasets, final-epoch reporting and the 500/200 epoch budget are local
multilabel adaptations; the original iBOT evaluator uses multiclass CE and
best validation accuracy. This is not claimed to reproduce CRISP exactly.
Launch into fresh output directories; protocol signatures reject old heads.
The whole-image recipes described above remain available explicitly.

The previous comparison used
`multilabel_recipe: asl224_lr001` (protocol version **7**), with effective
LR **0.001** from CRISP Appendix A.2, without batch scaling. All other
ASL/SGD settings below are retained. These settings are not claimed to recover
CRISP's unpublished loss, optimizer, pooling, or sampling details. Relaunches
use fresh output directories so the earlier LR 0.04 heads/results are not reused.

The earlier exploratory recipe remains available as
`multilabel_recipe: asl224` (protocol version **6**): SGD momentum **0.9**, effective
LR **0.04** without batch scaling, weight decay **0.01** on weights and biases,
and asymmetric loss with negative gamma **4**, positive gamma **0**, probability
clip **0.05**, detached focal weights, and mean reduction over globally known
labels. Whole-image resizing preserves image-level labels; there are no crops
or test-time augmentation. Epochs, batch size, feature pooling, initialization,
and cosine scheduling retain the settings above. This recipe obtained **89.56527**
full VOC mAP in the original-iBOT 224-pixel exploratory sweep. It was selected
on VOC validation and transferred unchanged to both checkpoints, COCO and VG;
it is **not** claimed to recover CRISP's unpublished protocol. The default for
other configurations remains `multilabel_recipe: bce`. Resume validation uses
the same task-specific protocol builder as the worker, including VOC shot count
and validation split, so completed few-shot tasks can be accepted correctly.

SGD with momentum 0.9 and zero weight decay, cosine decay without warmup, and
linear-head initialization follow the iBOT fallback. Multilabel masked BCE is
averaged over known image-class entries across all ranks. Results use the final
epoch; evaluation images do not select hyperparameters or checkpoints. Dataset
versions, membership, seed, BCE reduction, effective LR, preprocessing, and
the final-block extraction point are documented choices where CRISP is incomplete.
This is a reproducible benchmark, not a claim of exact CRISP score reproduction.

ImageNet keeps the original iBOT architecture-dependent pooling (last-four CLS
concatenation for ViT-S; final CLS plus mean patch features for ViT-B/L), training
RandomResizedCrop and evaluation Resize(256) + CenterCrop(224).

Earlier multilabel results are historical. New launches must use fresh output
directories: probe checkpoint signatures and completed-result protocol checks
reject incompatible reuse. `capi_classification.py` remains a legacy
comparison adapter; it is not selected by `imagenet_linear`.

ImageNet 1%, 10% and 100% k-NN continue to use fixed SimCLRv2/iBOT training banks,
final CLS features, temperature 0.07 and primary k=20, with full official
validation. The CRISP PDF does not specify k, temperature or exact subset
membership, so these remain explicit fallback choices. k-NN does not train a
linear classifier and therefore has no learning-rate/epoch requirement.

### Input manifests

ImageNet uses `<datasets-root>/imagenet/{train,val}` (the existing resolver also
accepts the other ImageNet directory spellings). Both must have the same 1,000
class directories. Multilabel tasks read:

```text
<datasets-root>/evaluation_manifests/pascal_voc.json
<datasets-root>/evaluation_manifests/coco.json
```

For VOC we select **VOC2012 classification train/val**, not segmentation/SBD:
5,717 training images and 5,823 validation images, with 20 ordered object
classes. Prepare its manifest from the already extracted official files:

```bash
python -m evaluation.prepare_voc_manifest --datasets-root dataset
```

The default source is `dataset/pascal_voc/VOCdevkit/VOC2012`; use `--voc-root`
for another location. The tool reads `ImageSets/Main/{train,val}.txt` and all
20 per-class annotation files, converts negative/difficult/positive labels to
`0`/`null`/`1`, records source-file hashes, and validates the manifest with the
evaluation loader before saving. Identical output is reusable; a different
existing manifest is not overwritten. Generated manifests live with the
locally prepared datasets rather than being committed to Git.

For COCO the selected baseline is **COCO 2017 train/val**: 118,287 training
images, 5,000 validation images, and 80 object categories. This is an explicit
experimental choice, not a verified reconstruction of CRISP's dataset split.
Use the official `train2017.zip`, `val2017.zip`, and
`annotations_trainval2017.zip` from the
[COCO downloads](https://cocodataset.org/#download). The HTTPS S3 path-style
endpoint `https://s3.amazonaws.com/images.cocodataset.org/` serves the same
official bucket without the certificate-name mismatch of the image hostname.
Archive SHA256 hashes can be checked against
[TensorFlow Datasets' checksum registry](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco.txt).

Downloading the ZIPs alone is not sufficient. After checksum verification,
extract the images and the two instance annotation files before generating the
manifest or submitting an evaluation:

```bash
mkdir -p dataset/coco/images
unzip -q -n dataset/coco/downloads/train2017.zip -d dataset/coco/images
unzip -q -n dataset/coco/downloads/val2017.zip -d dataset/coco/images
unzip -q -n dataset/coco/downloads/annotations_trainval2017.zip \
  'annotations/instances_train2017.json' 'annotations/instances_val2017.json' \
  -d dataset/coco
```

The local layout is:

```text
dataset/coco/images/train2017/*.jpg
dataset/coco/images/val2017/*.jpg
dataset/coco/annotations/instances_train2017.json
dataset/coco/annotations/instances_val2017.json
```

Generate and validate the evaluator's manifest with:

```bash
python -m evaluation.prepare_coco_manifest --datasets-root dataset
```

The generator orders classes by ascending original category ID (COCO IDs are
not contiguous). A class is positive if the image contains any instance of
that category, including crowd annotations, and negative otherwise. All
official images are retained, including images with no annotated instances.
The manifest records both annotation hashes, original category IDs, and this
label policy. Duplicate images, cross-split overlap, inconsistent vocabularies,
unknown annotation references, and missing image files fail validation. A
different existing manifest is never overwritten, and the new manifest is
published atomically only after complete validation. Captions, keypoints, stuff,
panoptic labels, and the unlabeled/test image archives are not needed here.

Use `--classification-manifests /path/to/manifests` to locate those files
elsewhere. Every file uses this schema (the example shows only two classes;
real files require 20 or 80 class names and that many label entries):

```json
{
  "dataset": "pascal_voc",
  "source": "Describe the exact dataset version, classification split and annotation source",
  "classes": ["aeroplane", "bicycle"],
  "splits": {
    "train": [
      {"id": "image-a", "image": "VOCdevkit/VOC2012/JPEGImages/image-a.jpg", "labels": [1, 0]},
      {"id": "image-b", "image": "VOCdevkit/VOC2012/JPEGImages/image-b.jpg", "labels": [0, 1]}
    ],
    "val": [
      {"id": "image-c", "image": "VOCdevkit/VOC2012/JPEGImages/image-c.jpg", "labels": [1, null]},
      {"id": "image-d", "image": "VOCdevkit/VOC2012/JPEGImages/image-d.jpg", "labels": [0, 1]}
    ]
  }
}
```

Image paths are absolute or relative to `--datasets-root`. `classes` defines
the order of every label vector. `1` means positive, `0` negative, and `null`
unknown/difficult. **Do not copy VOC's native -1/0/1 coding directly:** map
negative -1 to 0, difficult 0 to null, and positive 1 to 1. Classification
Pascal has 20 object classes; segmentation has 21 including background.
Classification images should come from the intended classification split,
not automatically from the segmentation/SBD lists.

Duplicate class names, duplicate images, train/val overlap, missing images,
invalid targets and training classes without positives/negatives fail before
training. Manifests supply the published inputs if available; this repository
does not claim that a newly invented split reproduces either paper.

## Running

For the classification tables, use the standard-GPU wrapper, which allocates
four GPUs, eight CPUs and 64 GiB with automatic checkpoint requeue:

```bash
sbatch slurm/evaluation_classification.sh config/evaluation_multilabel_region_vits200.yaml
sbatch slurm/evaluation_classification.sh config/evaluation_imagenet_region_vits200.yaml
```

Generated launch configs should set a persistent absolute output directory so a
requeue resumes the same probe. `evaluation.launch_slurm` also allocates four
GPUs for both classification groups.

Use Python 3.10/3.11 and the repository's CUDA requirements in production
(PyTorch 2.3's torch.compile does not support Python 3.12). Every evaluator
automatically launches one worker per visible NVIDIA GPU, including a proper
distributed process group on a single GPU. No GPU-count setting is required.
An explicit existing `torchrun` launch is respected. Zero GPUs fails clearly.

- CRISP linear classification requires four GPUs with 256 images per GPU,
  effective LR 0.001 for multilabel / iBOT-scaled LR 0.004 for ImageNet,
  and a single frozen linear probe. Small few-shot
  datasets and final batches can be smaller. DDP aggregates training gradients;
  validation shards count each evaluation image once, without padding.
- ImageNet k-NN distributes both feature extraction and validation queries.
  Each GPU holds the training bank; global top-1/top-5 count every validation
  image once.
- Segmentation distributes image extraction and CAPI's parameter sweep.
  Ordered features are replicated on CPU for the sweep using bounded tensor
  transfers. CAPI's final refit/scoring remains on rank 0, which alone writes
  the result JSON. Host memory requirements grow with worker count. The fixed
  eight-candidate search can use at most eight GPUs concurrently; extraction
  can use more. This follows the released distributed CAPI evaluator.

To restrict device use, set `CUDA_VISIBLE_DEVICES` before launching. When
resuming an unfinished linear probe, keep the same GPU count: its saved
protocol includes GPU/batch settings. Completed compatible scores remain reusable.

The main entrypoint is [`evaluation.py`](../evaluation.py); edit
[`config/evaluation.yaml`](../config/evaluation.yaml) to choose the checkpoint, checkpoint
key, architecture, dataset/manifest paths, output paths, seed, worker count,
segmentation extraction batch size, and individual evaluations. Numerical
probe recipes remain the fixed protocols described above.

```bash
python evaluation.py                    # Uses config/evaluation.yaml
python evaluation.py /path/to/run.yaml  # Uses another configuration
sbatch slurm/evaluation.sh              # Same config/evaluation.yaml on Slurm
sbatch slurm/evaluation.sh /path/to/run.yaml
```

The Slurm script contains scheduler resources and environment setup; it passes
the YAML filename to Python. It no longer chooses the checkpoint, dataset,
output directory, workers, or evaluation list. Slurm determines the allocated
GPUs: request one, eight, or another count using its `--gres` option. Python uses
the devices visible inside that allocation; it cannot acquire unallocated GPUs.
CUDA library paths for cuML are discovered from the active Python environment
by the evaluator, so the Slurm launcher follows the training launcher's layout.

### Weights & Biases

Set `wandb_mode`, `wandb_project`, `wandb_entity`, `wandb_run_name`,
`wandb_run_id`, and `wandb_resume` in `config/evaluation.yaml`. The corresponding
training settings live in `config/train.yaml`. Slurm contains no
W&B settings. `wandb_mode` accepts `online`, `offline`, or `disabled`.

Both training and evaluation read online authentication from `.wandb_key` at
the repository root, regardless of the YAML location. The file contains only
the API key; it is Git-ignored and is never included in YAML snapshots,
checkpoint arguments, or W&B run configuration. Offline/disabled modes do not
require a key. On another machine, place the key in the same root file.

Evaluation logs each task's scalar metrics under its own name, including
reused results, and records completion/failure in W&B. Training retains its
existing metric logging. To resume a W&B run, set `wandb_run_id` and
`wandb_resume: must` or `allow` in the YAML; specifying an ID alone defaults to
`must`. Resuming W&B does not itself resume model training or select a saved
evaluation directory; configure those paths separately.

### Selecting tasks and resuming evaluation

Each entry under `evaluations` is a YAML boolean. The supplied config enables
all 22 main-result tasks. Set unwanted tasks to `false`; omitted tasks are also
disabled. For example, replace that section with the following to run only
ImageNet 10% k-NN and VOC 1-shot classification:

```yaml
evaluations:
  imagenet_knn: true
  pascal_voc_1shot: true
```

Unknown keys/evaluations, duplicate keys, non-boolean switches and an empty
selection fail before launching workers. Only enabled multilabel tasks require
their manifests. Tasks run sequentially in the registry's fixed order.

Relative paths resolve against the YAML file's directory, independently of the
working directory. Absolute paths and `~` are supported. `output_dir: null`
creates a unique directory under the repository's `output/evaluation`;
`result_json: null` writes `full_evaluation.json` inside it. Each run saves
`evaluation_config.yaml` with resolved paths and all enabled/disabled switches.
The old `evaluation/full-evaluation` launcher has been replaced by `evaluation.py`.
Per-task modules in `evaluation/utils` are internal worker entrypoints.

The final JSON table distinguishes semantic segmentation, multiclass
classification and multilabel classification. Use `map_percent` for mAP on a
0-100 scale. Per-class AP is also retained. To resume, set `output_dir` to the
existing run directory, or run `python evaluation.py /path/to/run/evaluation_config.yaml`.
Completed results and probe checkpoints are accepted only if the model, data
and recipe match. Changing the selection alone preserves reusable results for
tasks that remain enabled.

## Sources

- CG-SSL: §4.2, Tables 1-2, reference [53] in the user-supplied 2025 PDF.
- CRISP: §4.2 and Appendix A.2, page 16 of the user's local 30-page
  `25314_Consistent_Region_Inform.pdf` (NeurIPS 2026 version).
- [CAPI classification evaluator](https://github.com/facebookresearch/capi/blob/98b4fa17ee8eec8810c17022df9a27a44845368b/eval_classification.py),
  [classification protocol](https://arxiv.org/html/2502.08769v1#A6.SS1), and
  [released defaults](https://github.com/facebookresearch/capi/blob/98b4fa17ee8eec8810c17022df9a27a44845368b/default_eval_config.yaml).
- [CAPI segmentation evaluator](https://github.com/facebookresearch/capi/blob/main/eval_segmentation.py)
  and [released dataset loader](https://github.com/facebookresearch/capi/blob/main/data.py), inspected 2026-09-08.
- [iBOT linear evaluator](https://github.com/bytedance/ibot/blob/main/evaluation/eval_linear.py)
  and [architecture-specific launcher](https://github.com/bytedance/ibot/blob/main/run.sh),
  used only for the explicitly identified choices not specified by CG-SSL/CRISP.

## Paper main-result tables

`config/evaluation.yaml` enables the 22 evaluations needed for Tables 1–5 of the
current Region Consistency draft. One launch reads exactly one `checkpoint`
and one `checkpoint_key`; the resulting `full_evaluation.json` groups all
selected task outputs for that checkpoint. The three ViT sizes require
separate launches with the checkpoint path changed between launches.

| Draft table | Evaluation names | Reported fields |
| --- | --- | --- |
| 1, segmentation | `ade20k_*`, `pascal_voc_knn`, `pascal_voc_linear`, `cityscapes_*` | `miou_percent` |
| 2, multilabel | `pascal_voc_{1,2,5}shot`, `pascal_voc_multilabel`, `coco_multilabel`, `visual_genome_multilabel` | `map_percent` |
| 3, ImageNet | `imagenet_knn_1pct`, `imagenet_knn`, `imagenet_knn_100pct`, `imagenet_linear` | `top1` |
| 4, correspondence | `spair_correspondence`, `navi_correspondence`, `scannet_correspondence` | viewpoint `d0/d1/d2/all`, rotation bins |
| 5, video segmentation | `davis_vos`, `youtube_vos_vos` | `j_and_f`, `j_mean`, `f_mean` |
| MOSEv2 submission export | `mose_vos` | indexed PNG masks and submission ZIP; scores pending external evaluation |

The paper's Tables 2–5 report ViT-S; Table 1 reports all three sizes. For
ViT-B/L, enable only the six Table 1 segmentation evaluations in a copy of
the YAML. Use a separate launch for each checkpoint. Source tables contain published baselines from other methods;
this suite computes the selected checkpoint's row and does not rerun those
external models.

For a run using the inputs currently available without dataset access approval,
use `config/evaluation_public.yaml` with `slurm/evaluation.sh`. It enables 21
tasks, saves into a stable output directory for resume after Slurm's time
limit, and leaves ScanNet disabled until its prepared test-pair dataset is available.
YouTube-VOS validation has full scoring annotations. MOSEv2 uses first-frame
initialization masks and exports predictions for manual server submission.
The full `config/evaluation.yaml` keeps all 22 tasks selected; its video preflight
checks scoring masks for DAVIS/YouTube-VOS and initialization masks plus official
video/frame metadata for MOSEv2 before any long probe begins.
`slurm/prepare_offline_evaluation_data.sh` fetches the public DAVIS, NAVI and
Visual Genome inputs. `slurm/prepare_youtube_vos_data.sh` can fetch the official
YouTube-VOS validation archive when Google Drive permits it, but its completion
alone does not establish that the scoring masks are present.

### Added data inputs

Visual Genome uses the **VG500** 500-category benchmark. Supply
`<datasets_root>/evaluation_manifests/visual_genome.json` with the same JSON
schema as VOC/COCO manifests: `dataset: visual_genome`, a nonempty `source`,
500 ordered `classes`, and `train`/`val` rows containing `image`, `id`, and
either 500-entry `labels` vectors (`0`, `1`, or `null`) or sparse
`positive_indices` lists (other classes are negative). The exact VG500 split and image labels used by CRISP were not released in
its paper. A reproducible public choice is the SSGRL VisualGenome-500 release:
its `train_list_500.txt`, `test_list_500.txt`, and
`vg_category_500_labels_index.json` are available in SSGRL's `data/VG` folder.
Place the original Visual Genome images under
`<datasets_root>/visual_genome/VG_100K` and `VG_100K_2`, then run:

```bash
python -m evaluation.prepare_visual_genome_manifest \
  --datasets-root /path/to/datasets \
  --annotations-dir /path/to/SSGRL/data/VG
```

The preparer maps the public **test** list to the manifest's evaluation split,
uses all 500 class indices in their released order, and hashes the three
annotation files. It fails on missing images or train/test overlap. Reuse the
same generated manifest for every checkpoint. The classifier uses the frozen
224-pixel, 200-epoch, global-batch-1024 linear protocol of CRISP's COCO probe.
This public split is a concrete baseline, not a claim that it reproduces
CRISP's unpublished VG500 image lists exactly.

Correspondence inputs use the released Probe3D layouts:

```text
<datasets_root>/SPair-71k/{PairAnnotation,ImageAnnotation,JPEGImages,Segmentation}
<datasets_root>/navi_v1/<object>/{wild_set,multiview_*}/...
<datasets_root>/scannet_test_1500/{test.npz,intrinsics.npz,<scene>/...}
```

After extracting NAVI, run `python -m evaluation.prepare_navi
<datasets_root>/navi_v1 --workers 8` (or `sbatch
slurm/prepare_navi_downsampled.sh`). The released Probe3D reader requires
`downsampled_` RGB/depth files made with its 1024-pixel resize recipe; the
original NAVI archive has only the source files.

The correspondence iBOT adapter uses raw final-block patch tokens before
the final LayerNorm, matching Probe3D's released `evals/models/ibot.py`.
SPair applies L2 normalization before keypoint sampling; NAVI applies it
after bicubic feature interpolation to the geometry grid. This differs from
the normalized block features used by the video and segmentation probes.
The SPair test split uses 800-pixel images without bounding-box crop,
PCK@0.1, and at most 200 seeded pairs per category and viewpoint
level. NAVI uses its in-the-wild test pairs, 512-pixel bbox crop, 1,000
ratio-ranked correspondences, and 2-cm 3D recall in four rotation bins.
ScanNet uses released test pairs at 480 x 640, 1,000 correspondences, and
10-pixel reprojection recall on Probe3D's quarter-resolution geometry grid.
Correspondence results also record the actual selected pair identities.
Controlled feature comparisons can set `correspondence_feature_variant` to
`final_norm_standardized`, `projection`, `projection_softmax`, or
`concat_4_6_8_12` in the evaluation YAML. The default remains `raw_final`.
These alternatives retain the released pairs and matching protocol, but are
feature ablations rather than the official Probe3D feature recipe.
The standardized variant fits channelwise `StandardScaler` on normalized
patch features from CAPI's seeded 90% VOC training subset at 256 pixels for
patch size 16, then freezes it for every correspondence test dataset. It
never fits on correspondence test images. Saved scaler statistics and training
image indices identify the calibration inputs.
Projection variants restore the checkpoint's trained teacher patch MLP and
prototype layer with strict loading. `projection` uses its output logits;
`projection_softmax` applies plain per-patch softmax, without centering or
Sinkhorn, at `correspondence_softmax_temperature` (default 1.0).
`concat_4_6_8_12` concatenates the raw outputs of those four one-based blocks,
without final LayerNorm, following iBOT's linear segmentation feature recipe.
Upstream pair sampling depends on filesystem enumeration, so seed equality
alone cannot establish equality with CRISP's unpublished sampled pairs.
All three results include split and threshold metadata. Probe3D's published
10-pixel cutoff is measured on that quarter-resolution ScanNet grid; the
current draft should state this if those numbers are used in Table 4.
Probe3D documents a public LoFTR download of `scannet_test_1500`, which
contains the RGB, depth and poses needed here without downloading the full
ScanNet release: see [its dataset instructions](https://github.com/mbanani/probe3d/blob/main/data_processing/README.md#scannet-correspondence-test-split).

Video inputs use standard validation layouts:

```text
<datasets_root>/davis2017/{ImageSets/2017/val.txt,JPEGImages/480p,Annotations/480p}
<datasets_root>/youtube_vos_2019/valid/{JPEGImages,Annotations}
<datasets_root>/MOSEv2/valid/{JPEGImages,Annotations}
<datasets_root>/MOSEv2/meta_valid.json
```

For the local YouTube-VOS 2019 and MOSEv2 archives in `datasets_root/raw`, run
`sbatch slurm/prepare_downloaded_vos.sh`. This verifies the published MOSEv2
SHA-256 checksums, extracts the nested YouTube-VOS releases, and joins the
two-part all-frame archives before extraction. It writes YouTube-VOS to
`youtube_vos_2019/` and MOSEv2 to `MOSEv2/`, retaining the original downloads.
The all-frame YouTube-VOS RGB releases, released test ground truth, and scoring
program are kept separately from the validation split. MOSEv2 sample predictions
are never installed as ground truth. The preparation inventory is written to
`downloads/prepared_vos/preparation_report.json`; its `offline_scoring_ready`
field distinguishes extraction completeness from availability of scoring masks.
MOSEv2 is kept separate from MOSEv1 and its results are labeled `MOSEv2 val`.
The `mose_vos` worker now exports predictions instead of computing offline metrics.

Alternate capitalization for the DAVIS/YouTube-VOS top-level directory is accepted.
YouTube-VOS may use `val/` instead of `valid/`; its `meta.json` and complete
validation scoring masks must be installed for offline J/F scoring. The
preflight rejects the usual first-frame-only validation release. For
YouTube-VOS,
provide a standard `ImageSets/val.txt` list when possible; otherwise all
sequence directories are evaluated and the selected names are recorded.
MOSEv2 requires the complete validation RGB split listed in `meta_valid.json`
and its first-frame masks. Later masks are not required or used as ground truth.
Video propagation uses 480 x 480 inputs, the mean of the last four normalized
patch-token blocks, first-frame reference plus seven past frames, radius 12,
cosine temperature 0.1 and top-5 affinity, consistent with CRISP's feature
choice and DINO's released propagation defaults. The boundary/region metrics
come from the official DAVIS evaluator. Initial masks and DAVIS final-frame
masks are excluded from scoring; labeled YouTube-VOS frames are scored.
For YouTube-VOS, a new object’s first annotated appearance becomes an
additional reference and is excluded from that object’s own score. The
YouTube-VOS score uses the official metadata's object frame lists, the
360-pixel boundary protocol, and the mean of seen/unseen J/F groups. CRISP does
not publish its exact video split lists or this detail of new-object handling,
so published-score reproduction is not guaranteed; retain the saved
split/protocol metadata with comparisons.

For MOSEv2, each run writes native-resolution, palette-indexed PNG masks to
`<evaluation-output>/mose_vos/Annotations/<video>/<frame>.png`. Object IDs and
the initialization palette are preserved. The initialization mask is included
as frame zero, followed by predictions for every RGB frame, including the last.
The worker also creates
`<evaluation-output>/mose_vos/mosev2_valid_submission.zip`, containing
`<video>/<frame>.png` directly at the ZIP root, matching the official sample
submission. Upload this ZIP manually to the
[MOSEv2 evaluation server](https://www.codabench.org/competitions/10062/).
The result JSON records export paths/counts and
`metrics_status: pending_external_evaluation`; J/F entries are `null` until
scored externally. A completed export is reused only while its submission ZIP
still exists. Submission is not automatic.

## Slurm resources and combined suites

Use `./.conda-env/bin/python -m evaluation.launch_slurm config/evaluation_region_vits200.yaml`
to split a full suite into independent jobs. `--dry-run` previews allocations:

| Group | GPUs | CPUs | RAM | Time limit |
| --- | ---: | ---: | ---: | --- |
| Segmentation | 1 | 16 | 128 GiB | 48 hours |
| ImageNet linear | 4 | 24 | 64 GiB | 72 hours |
| Other classification | 4 | 24 | 64 GiB | 24 hours |
| Correspondence/video | 1 | 8 | 32 GiB | 72 hours |

All groups use the same checkpoint, datasets and seed, with separate outputs.
A dependent CPU job merges their results into the suite's `full_evaluation.json`.
`python -m evaluation.merge_results <suite-directory>` refreshes partial results.
Groups requeue ten minutes before their limits. Completed evaluations are reused;
unfinished linear heads resume optimizer/scheduler state. An interrupted dense
fit or video/correspondence task may repeat its unfinished task.

The generic `sbatch slurm/evaluation.sh <config>` remains available for one
allocation (four GPUs, 128 GiB, 72 hours). Its offline segmentation workers see
only one GPU. The grouped launcher avoids reserving the other GPUs for these
tasks. Selected dataset splits and local multilabel BCE/AP/final-epoch choices
remain unchanged and are recorded in result metadata.

## Native CRISP depth and surface normals

`crisp_native_dense.py` snapshots the supplied CRISP/Probe3D repository and
uses its depth and surface-normal training entry points, loaders, DPT heads,
losses and metrics. Its region ViT-S adapter supplies raw blocks 3/6/9/12 as
four separate DPT inputs. The backbone remains frozen; decoders train for
10 epochs with the native optimizer and schedule. Saved CRISP batch sizes
are preserved: depth 2, surface normals 8, on one GPU.

Prepare a run using the supplied code and data:

```bash
python evaluation/crisp_native_dense.py prepare --root /absolute/path/to/run
sbatch slurm/evaluation_crisp_native_dense.sh /absolute/path/to/run depth_nyu
sbatch slurm/evaluation_crisp_native_dense.sh /absolute/path/to/run depth_navi
sbatch slurm/evaluation_crisp_native_dense.sh /absolute/path/to/run snorm_nyu
sbatch slurm/evaluation_crisp_native_dense.sh /absolute/path/to/run snorm_navi
```

Defaults point to `/mnt/fast/nobackup/scratch4weeks/jg02228/probe3d` and the
region ViT-S continuation-epoch-200 teacher checkpoint. Override these with
`--crisp-root` and `--checkpoint`. Dataset files are linked, rather than copied.
NYU uses all 30,914 valid GeoNet training instances and 654 labeled test
images at 480×480. NAVI uses multiview training and wild testing, retaining
the supplied loader's every-fourth selection (2,024/555 images) at 512×512.

Every task checks source/checkpoint hashes, data identities, representative
samples, and decoder forward/backward at its actual batch size before training.
The sole training-file fix handles final checkpoint serialization on a single
GPU; training and metric computations are unchanged. The supplied code's
unseeded training behavior is also preserved despite its seed configuration.
Slurm requests RTX8000/A100 GPUs. NYU's labeled pickle is approximately 8 GiB;
the default allocation includes 32 GiB RAM and 48 hours.

Refresh metrics using the copied environment:

```bash
/mnt/fast/nobackup/scratch4weeks/jg02228/probe3d/env_probe3d/bin/python \
    /absolute/path/to/run/native_dense.py collect --root /absolute/path/to/run
```

The run records source hashes, the exact checkpoint identity, resolved configs,
and selected splits. Results appear in `results.json` and `results_summary.md`;
native logs and decoder checkpoints remain under `tasks/*/source/`.

## SPair-71k scale and visibility robustness

`spair_nuisance.py` evaluates original iBOT ViT-S/16 and the region epoch-200
teacher on all 12,234 pairs in `Layout/large/test.txt`. It reuses the supplied
CRISP `compute_errors` function and preprocessing at 800 pixels: raw final-block
features, L2 normalization before source-keypoint sampling, cosine argmax,
no bounding-box crop, no feature head, no augmentation. Native square padding
and integer keypoint rescaling remain unchanged.

The primary PCK follows the [official HPF evaluator](https://github.com/juhongm999/hpf/blob/master/model/evaluation.py):
errors at most 0.1 times the longer target bounding-box side count as correct;
PCK is calculated per pair, then averaged over pairs. This differs from
Probe3D's category average of pooled keypoint correctness, which is also saved
as a secondary metric. These full-split scores need not equal earlier sampled
Probe3D results. The subsets follow Table 4 of the
[SPair paper](https://arxiv.org/abs/1908.10543), independently grouping each
annotated factor while leaving the other factors unrestricted. This is not
the paper's Table 5 analysis that holds other factors fixed.

Viewpoint/scale codes 0/1/2 mean easy/medium/hard. Truncation and occlusion
codes 0/1/2/3 mean none/source/target/both. Preparation checks these directions
against image-level annotations, validates all common keypoints, and verifies
all 14 subset counts against the publication. Source code, annotations, ordered
pairs and checkpoint identities are frozen and recorded in the run manifest.

```bash
python evaluation/spair_nuisance.py prepare --root /absolute/path/to/run
sbatch --array=0-1 slurm/evaluation_spair_nuisance.sh /absolute/path/to/run
python /absolute/path/to/run/spair_nuisance.py collect --root /absolute/path/to/run
```

Each model uses one GPU, four CPUs, 16 GiB RAM and up to 12 hours. Before
evaluating, the GPU worker checks checkpoint/source identities and tests real
matching on directional visibility subsets. Per-pair errors are retained in
`MODEL/pairs.jsonl` and allow restarting at the last completed pair. Final
metrics appear in `MODEL/results.json`; collecting produces combined
`results.json`, `results_summary.md`, and `table_rows.tex` in the run root.

## CRISP correspondence with normalized backbone features

`crisp_native_features.py` compares two feature choices for the region epoch-200
teacher: final-block patch tokens after the learned final LayerNorm (the VOC
feature extraction point), and the average of blocks 9–12 after applying that
LayerNorm separately to each block (the video feature extraction rule).
The feature width remains 384. The projection head and softmax are not used.
VOC's classifier-specific StandardScaler is not added to correspondence;
the native correspondence code retains its L2 normalization and matching.

The runner freezes the previous CRISP-native source tree, configs, checkpoint
identity, and exact SPair/NAVI/ScanNet pair lists. It changes only the backbone
adapter and pins the pair order for comparison with the raw-block baseline.
GPU preflight checks normalized features against explicit block-by-block
extraction and exercises the 800-pixel SPair input before evaluation.

```bash
python evaluation/crisp_native_features.py prepare --root /absolute/path/to/run
sbatch --array=0-5 slurm/evaluation_crisp_native_features.sh /absolute/path/to/run
python /absolute/path/to/run/runner.py collect --root /absolute/path/to/run
```

Tasks 0–2 evaluate final LayerNorm features on SPair, NAVI, and ScanNet;
tasks 3–5 evaluate the last-four average in the same order. Each uses one GPU,
four CPUs, 16 GiB RAM, and up to 12 hours. Logs and preflight reports are under
`VARIANT/DATASET/`; collecting writes `results.json` and `results_summary.md`.
# NeCo Table 5: SPair-71k multiview feature consistency

`spair_neco.py` implements the [NeCo Table 5](https://arxiv.org/html/2408.11054v3#S4.T5)
protocol using the cited official Probe3D `compute_errors` function unchanged.
Images use the Probe3D square padding/preprocessing at **224 × 224** instead of
800; frozen teacher **raw final-block** patch tokens are L2-normalized before
cosine matching. There is no projection head, softmax, foreground masking,
training-set scaler, feature averaging, or test-time augmentation.

The table caption specifies **Recall@0.01**. The primary report pools correct
keypoints within each of the 18 categories and averages category recalls,
following Probe3D. Recall@0.1 is saved separately because that is the threshold
hardcoded in the released Probe3D evaluator. Errors use the target bounding-box
longest side as their scale. The implementation also saves the official
SPair/HPF per-pair PCK average with its inclusive threshold convention.

NeCo does not release its SPair evaluator or sampled pairs. Therefore this is
a reconstruction from its stated settings and cited Probe3D source, rather
than a verified reproduction of its exact table. Both checkpoints use the same
pinned seed-20 Probe3D sample (up to 200 pairs per category), followed by the
remaining pairs of the official 12,234-pair test layout. Reports for the sampled
and full test sets are kept separate. The sampled report becomes available
before the full run finishes.

```bash
python evaluation/spair_neco.py prepare --root /absolute/path/to/new/run
sbatch --array=0-1 slurm/evaluation_spair_neco.sh /absolute/path/to/new/run
python /absolute/path/to/new/run/runner.py collect --root /absolute/path/to/new/run
```

Preparation freezes the source, annotations, ordered pair lists, checkpoint
identities, runner, and launcher in the output directory. Each GPU job checks
source/checkpoint hashes, validates raw-block feature equivalence, and records
one result per pair in a journal that supports resuming interrupted evaluations.
`results_sampled.json`, `results_full.json`, `results_summary.md`, and
`table_rows.tex` contain the final values and protocol provenance.


## Unsupervised object discovery (DINOv3 / TokenCut)

`config/evaluation_object_discovery.yaml` evaluates official DINO v1 ViT-S/16,
original iBOT ViT-S/16, and the region200 teacher using DINOv3 Sec. 6.1.4 /
Appendix D.4. Run this independently of the classification/video suite:

```bash
.conda-env/bin/python -m evaluation.object_discovery prepare \
  --config config/evaluation_object_discovery.yaml --root /absolute/new/run
sbatch --array=0-8 slurm/evaluation_object_discovery.sh /absolute/new/run
.conda-env/bin/python -m evaluation.object_discovery collect --root /absolute/new/run
```

Data requirements: VOC2007 detection **trainval** (5,011 images), VOC2012
**trainval** (11,540), and the official fixed COCO20K list (19,817) with
`coco/annotations/instances_train2014.json`. Existing COCO2017 JPEGs are resolved
by the original COCO numeric image IDs; no 2017 split is substituted for COCO20K.
Preparation validates and decodes every image, checks annotation dimensions,
hashes images/checkpoints/annotations, and creates a frozen source snapshot.

All backbones use final LayerNorm patch outputs, ImageNet normalization, native
image resolution with official TokenCut zero padding, FP32 without AMP/TF32,
and unchanged official TokenCut normalized cut. The extended sweep covers
0.00 through 0.95 in steps of 0.05; the published 0.00--0.40 range is also
summarized separately. VOC difficult and truncated boxes
are included; COCO crowd boxes are excluded as in the official TokenCut release.
Every image remains in the denominator, including any images with no noncrowd
boxes. CorLoc is the percentage of images with a predicted box having IoU > 0.5
with any eligible GT box. The >= 0.5 convention is also recorded separately.

Each array task handles one model/dataset, saving all twenty predictions per image
in an atomic SQLite transaction. Requeues resume at the first uncommitted image.
`<model>/<dataset>/results.json` contains the whole threshold sweep; only a
completed dataset receives a final best result. `results_summary.json` is refreshed
by `collect`. The best threshold is selected per model/dataset, never per image.

The public DINOv3 repository contains no object-discovery evaluator at the
inspected commit. This extends its **published** threshold sweep with official TokenCut;
unspecified implementation details are documented in the run manifest and
[evaluation/vendor/tokencut/README.md](vendor/tokencut/README.md).


## NeCo frozen-feature comparisons and semantic region retrieval

See [neco_protocol.md](neco_protocol.md) for pinned authors' code, paper/config
discrepancies, feature definitions and data audit rules. Both original iBOT and
Region ViT-S/16 continuation epoch 200 are evaluated. The NeCo sweep contains
64 dense retrieval tasks (VOC/ADE, published fractions and five partial-subset
seeds), 16 clustering tasks (four datasets, class-count K and K=500, five
internal seeds), and four COCO linear probes.

Prepare a new frozen sweep before submitting the task arrays:

```bash
.conda-env/bin/python -m evaluation.neco_benchmarks --prepare \
  --config config/evaluation_neco_ibot_vs_region200.yaml \
  --root output/evaluation/neco_NEW_RUN
```

`slurm/prepare_neco_data.sh` prepares official annotations without duplicating
images. VOC and COCO GPU arrays must depend on the corresponding successful
preparation task, and the runner requires its completed audit JSON.
`slurm/evaluation_neco.sh` executes one frozen comparison task per array index.
Use 128 GiB host RAM for the published 10,240,000-entry retrieval bank;
clustering and COCO linear probes request 32 GiB. Each uses one GPU.

`evaluation/collect_neco.py --root RUN_ROOT` reports all seeds and only computes
final aggregate results when each group is complete. A dependent collector is
submitted for the current sweep.

`slurm/evaluation_semantic_region_retrieval.sh` evaluates original iBOT and
Region200 on disjoint VOC2012 train-gallery / validation-query regions. This
is a separate custom mask-conditioned retrieval analysis, not a NeCo score.
It reports macro/micro mAP and Recall@1/5/10, with exact region identities and
split hashes retained for comparisons.
