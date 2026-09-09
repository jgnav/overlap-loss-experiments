# Evaluation protocols

Each evaluation has one implementation and one fixed recipe. There are no
CG-SSL/CAPI/CRISP mode switches. The main references are CG-SSL and CRISP;
CAPI supplies the segmentation implementation they cite.

## Selected segmentation protocol

Segmentation calls the vendored official CAPI evaluator, pinned to revision
`98b4fa17ee8eec8810c17022df9a27a44845368b`. Its classifiers, hyperparameter
selection, refitting and scoring are copied from upstream; local adapters
handle dataset paths, final normalized patch features, logging and JSON output.
See [vendor provenance and integration changes](vendor/capi/README.md).

**CRISP's 256-patch-token convention** is applied dynamically: the checkpoint's
convolution kernel determines patch size, and input resolution is 16 times
that size. Patch size 14 uses **224 x 224**; patch size 16 uses **256 x 256**.
Both produce a 16 x 16 grid. Positional embeddings retain their checkpoint grid
and interpolate during inference. Compatible iBOT-style ViT-S/B/L checkpoints
are supported; this does not promise compatibility with arbitrary architectures
(e.g. register tokens or different transformer blocks). Unsupported patch sizes
fail explicitly. Classification remains 224 x 224.

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
not used. Both retain the same CAPI-style holdout/probes and CRISP's 256-token
resolution. ImageNet classification uses official train/val with matching
1,000-class vocabularies; k-NN uses a seeded stratified 10% training bank and
linear probing uses all training images. VOC multilabel classification remains
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

## Full-data classification

The three added evaluations cover these classification tasks from CG-SSL Table 2:

| Evaluation | Training data | Epochs | Metric |
| --- | --- | --- | --- |
| `imagenet_linear` | Full ImageNet-1K train | 200 | Top-1/top-5 (%) |
| `pascal_voc_multilabel` | Explicit Pascal classification train split, 20 classes | 500 | mAP |
| `coco_multilabel` | Explicit COCO classification train split, 80 classes | 200 | mAP |

CRISP A.2 specifies 224 x 224 inputs, four GPUs, batch size 256 per GPU, learning
rate 0.001, and these epoch counts. The frozen backbone remains in evaluation
mode; only a linear layer is trained. The existing `imagenet_knn` remains the
10% reference-bank evaluation used in the original repository's CRISP ablations.
It is not paired with the full-data ImageNet linear score as if both used the
same amount of labeled training data. No extra low-shot evaluation was added.

### Details not established by the papers

CG-SSL does not specify a complete classification training recipe. CRISP adds
the settings above, but still omits optimizer/schedule, pooling, augmentation,
exact dataset splits/vocabularies, AP variant, and checkpoint selection. This
implementation therefore **cannot claim exact published-score reproduction**.
The full protocol and unresolved choices are saved in every result JSON:

- iBOT-derived pooling: concatenate the last four normalized CLS tokens for
  ViT-S; for ViT-B/L concatenate the final CLS token and mean final patch token.
- iBOT-derived optimizer: SGD, momentum 0.9, no weight decay, epoch-wise cosine
  decay to zero, no warmup. The actual initial learning rate is 0.001; the code
  does not apply iBOT's extra batch-size rescaling to CRISP's stated rate.
- iBOT-derived transforms: train RandomResizedCrop(224, bilinear) and horizontal
  flip; validation shorter-side resize to 256 (bicubic), center crop to 224;
  ImageNet normalization for both. Features are recomputed from augmented
  images during probe training, not cached from one deterministic crop.
- Multiclass cross entropy; multilabel BCE over known labels only. Unknown
  labels do not contribute gradients or AP. Report non-interpolated per-class
  average precision and average over the fixed vocabulary. A class with no
  validation positives gets AP=0 and is explicitly listed in the result.
- Evaluate the final epoch. Do not select the best epoch on the reported
  validation set. Probe checkpoints support resuming the same recipe.

These are documented implementation choices, not additional alternative
protocols. Author-supplied evaluation code/configs would be needed to verify
them. The full-data multilabel manifests below deliberately make all data
definitions explicit rather than guessing VOC/COCO versions or splits.

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

Use Python 3.10/3.11 and the repository's CUDA requirements in production
(PyTorch 2.3's torch.compile does not support Python 3.12). All
classification probes require four visible GPUs; dense probes require one.

```bash
python evaluation/full-evaluation /path/to/checkpoint.pth \
  --datasets-root /path/to/datasets
```

The full command runs ten evaluations and preflights the two multilabel
manifests before expensive work begins. Select tasks without adding alternative
protocols:

```bash
python evaluation/full-evaluation /path/to/checkpoint.pth \
  --datasets-root /path/to/datasets \
  --evaluations imagenet_linear pascal_voc_multilabel coco_multilabel

python -m evaluation.utils.imagenet_linear /path/to/checkpoint.pth \
  --datasets-root /path/to/datasets
```

The final JSON table distinguishes semantic segmentation, multiclass
classification and multilabel classification. Use `map_percent` for mAP on a
0-100 scale. Per-class AP is also retained. To resume, pass the same `--output-dir`;
the probe checkpoint is accepted only if the model, data and recipe match.

## Sources

- CG-SSL: §4.2, Tables 1-2, reference [53] in the user-supplied 2025 PDF.
- CRISP: §4.2 and Appendix A.2, page 16 of the user's local 30-page
  `25314_Consistent_Region_Inform.pdf` (NeurIPS 2026 version).
- [CAPI segmentation evaluator](https://github.com/facebookresearch/capi/blob/main/eval_segmentation.py)
  and [released dataset loader](https://github.com/facebookresearch/capi/blob/main/data.py), inspected 2026-09-08.
- [iBOT linear evaluator](https://github.com/bytedance/ibot/blob/main/evaluation/eval_linear.py)
  and [architecture-specific launcher](https://github.com/bytedance/ibot/blob/main/run.sh),
  used only for the explicitly identified choices not specified by CG-SSL/CRISP.
