# Evaluation protocols

Each evaluation has one implementation and one fixed recipe. There are no
CG-SSL/CAPI/CRISP mode switches. The main references are CG-SSL and CRISP;
CAPI supplies the segmentation implementation they cite.

## Selected segmentation protocol

The user selected **CRISP's 256-patch-token convention** from Appendix A.2 of
the local `25314_Consistent_Region_Inform.pdf` (page 16): ViT/16 inputs are
**256 x 256**, producing a 16 x 16 token grid. This differs from the 224 x 224
default in the CAPI protocol cited by CG-SSL. Classification remains 224 x 224.

ADE20K, PASCAL VOC 2012 and Cityscapes use frozen final-block patch features,
train-only StandardScaler, a seeded 10% training holdout for selecting probe
hyperparameters, and refitting on train plus holdout before evaluating the
official validation set. k-NN tests k in {1, 3, 10, 30} with cosine/L2 distance;
linear segmentation uses CAPI's cuML logistic-regression sweep. The paper-style
mIoU percentage is `metrics.miou_percent`.

VOC `trainaug` now follows the **released CAPI loader literally**:
original VOC training list, then SBD training list, then SBD validation list.
List order and repeated IDs are preserved. Original entries use original PNG
masks; SBD entries use SBD MAT masks. The evaluation set is original VOC val.

**Upstream limitation:** that released loader explicitly states that it does
not reproduce CAPI's paper results. Its concatenation can retain both duplicate
images and official validation images appearing in SBD. Reproducing the loader
does not establish the image split used for CG-SSL/CRISP's published numbers.
Results record duplicate counts, official-val overlap, and ordered image/mask
pair hashes under `dataset_manifests`. Interpret these as reproduction of the
released loader, including its limitations, rather than a clean held-out VOC
benchmark. Do not erase these metadata when reporting scores.

Changes to resolution, dataset list construction, seed, source code, or manifest
files invalidate old caches/results. Incompatible dense caches are recomputed.
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
