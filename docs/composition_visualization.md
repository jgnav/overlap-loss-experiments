# Controlled A + B composition figure

`composition_visualization.py` produces one 300-DPI PNG from a full iBOT
checkpoint, a numeric COCO image ID, and your datasets root. The image is found
recursively and its mask is generated from local COCO instance annotations.
It does not download data or checkpoints. It defaults to CPU.

```bash
python composition_visualization.py
```

All inputs and experimental settings are hard-coded near the top of the script. Configure
these before running:

```python
CHECKPOINT = Path("/path/to/checkpoint.pth")
COCO_IMAGE_ID = 240684
DATASETS_ROOT = Path("/path/to/datasets")
CONCEPT_A = "Dog"
CONCEPT_B = "Car"
OBJECT_A_COLOR = (255, 0, 0)
OBJECT_B_COLOR = (0, 255, 0)
BACKGROUND_COLOR = (0, 0, 0)
REFERENCE_A = [
    ReferenceRegion("/data/dog1.jpg", "/data/dog1_mask.png", (255, 0, 0)),
    ReferenceRegion("/data/dog2.jpg", "/data/dog2_mask.png", (255, 0, 0)),
]
REFERENCE_B = [
    ReferenceRegion("/data/car1.jpg", "/data/car1_mask.png", (0, 255, 0)),
    ReferenceRegion("/data/car2.jpg", "/data/car2_mask.png", (0, 255, 0)),
]
```

Reference colors are specified separately for each region; they need not match
the displayed image's colors. References must be independent of the displayed
image. The script rejects identical decoded images, including lossless copies;
you must also keep near-duplicates and alternate crops out of calibration.
There is no same-image fallback. Several reference regions per concept are
preferable to one. Reference A and B must have the intended semantic meaning;
mask colors alone cannot establish that meaning.

The root must contain extracted COCO images and `instances_*.json` annotation
files (for example `coco/train2017` and `coco/annotations`). No precomputed PNG
mask is needed. Install `pycocotools` in the active environment for faithful
polygon and compressed/uncompressed RLE decoding. The script recursively finds
the image ID in annotation files and locates its recorded image filename. Missing
or ambiguous matches fail with an actionable error; narrow the root if it contains
multiple COCO releases or duplicate copies.

`CONCEPT_A/B` match COCO category names, case-insensitively. The largest non-crowd
instance per category is selected. With generic `"A"`/`"B"`, selection instead uses
the two largest instances of distinct categories, breaking area ties by annotation
ID. Set `COCO_ANNOTATION_A/B` to exact instance IDs to override automatic selection.
Use configured category names to keep selection consistent with your independent
reference concepts. Unrelated instances are excluded. Pixels shared by both selected
masks are excluded from both, avoiding arbitrary overlap ownership. Selection,
annotation source, category names, and instance IDs are printed and saved in JSON.
The generated RGB mask is saved alongside the figure using `OBJECT_A/B_COLOR`.

Independent reference masks still use the paths/colors configured at the top of
the script. They may be RGB, palette PNG, or grayscale IDs (ID 7 becomes RGB
`(7, 7, 7)`). Use exact labels and matching image/mask dimensions.
The image is resized with bicubic interpolation, the mask with nearest-neighbor,
and both are padded at the right/bottom to a complete patch grid. No object is
removed by a center crop.

## Representation and figure

The script uses the EMA teacher's final spatial tokens through its actual patch
projection head, supports the repository's shared/separate head checkpoints,
and does not subtract the teacher center. `PATCH_PURITY = 0.9` selects patches
with at least 90% of their area in the object mask. Every accepted patch has
weight one. This object-purity threshold is separate from the training crop
intersection threshold.

`NORMALIZATION_OVERRIDE = None` reads `region_normalization` from checkpoint
arguments, defaulting to uncentered softmax for original iBOT checkpoints.
`TEMPERATURE_OVERRIDE = None` reads `region_temp`, defaulting to `0.1`.

- Softmax operates independently on each patch.
- Centering restores the saved iBOT patch center (`center2`) and applies the
  checkpoint's `teacher_patch_temp` to the raw teacher patch logits. This is
  the same teacher target representation used by ordinary iBOT.
- Sinkhorn uses the training implementation with three iterations. All accepted
  calibration patches from both concepts share one calibration assignment bank.
  The displayed image's accepted A+B patches form a separate bank. Assignments
  are computed once and frozen for the region means and five mixture bars.
  They are not refit separately for A, B, or each mixture, since balancing each
  pure region could erase the very contrast being measured. Sinkhorn results
  depend on this recorded assignment scope and should not be interpreted as
  invariant to a change in batch composition.
- `raw_logits` checkpoints are rejected because signed L2-normalized vectors
  do not define probability mass. An explicit `NORMALIZATION_OVERRIDE = "softmax"`
  is allowed, but then the figure visualizes that override, not the checkpoint's
  trained raw-vector region representation. The override is recorded in JSON.

Pure-region reference means are averaged **equally across regions** to obtain
mu_A and mu_B. The A set contains up to `TOP_K` positive entries of mu_A-mu_B;
the B set uses the negative entries in reverse order. Ties remain in other.
The sets are disjoint, so A, B, and all other components form a true partition
of probability mass. The script fails if calibration provides no contrast;
it does not invent component associations or force separation.

The PNG contains:

1. Source image with object-mask boundaries.
2. Patch grid showing the mask-selected A and B patches.
3. Patch grid classified by the largest A-associated, B-associated, or other mass.
   Rejected patches remain unclassified; the pooled A+B mass appears below it.
4. Prototype fingerprints for A, B and A+B using the fixed reference-selected
   dimensions and one common probability color scale, without per-row scaling.
5. Three 100% stacked bars for all accepted A, B and A+B patches.
6. Five controlled patch-count mixtures: 100/0, 75/25, 50/50, 25/75 and 0/100.
7. The mean-pooling formula and protocol summary.

All dimensions remain intact during computation; only the fingerprint panel
selects individual components for display. No PCA, UMAP, clustering, or learned
projection is performed. Bars preserve measured mass, including a dominant
“other” component when present. A/B-associated components are empirical latent
associations, not guaranteed semantic prototype labels.

## Controlled mixtures and interpretation

The mixture count is the largest multiple of four no greater than
`min(MIXTURE_PATCHES, accepted_A_count, accepted_B_count)`. At least four pure
patches per object are required; the script never silently lowers purity or
samples patches with replacement. A fixed seed establishes one patch ordering
per concept. Mixtures use prefixes of those orderings, keeping total patch count
constant and achieving the requested ratios exactly. A-only and B-only mixture
endpoints may therefore differ from the all-patch region means above them.

These are controlled **patch subsets in a fixed image**, not physically pasted
images or new model forwards. The all-patch union obeys
`r_AB = (n_A*r_A + n_B*r_B)/(n_A+n_B)` by construction; the maximum absolute
residual is recorded. This illustrates what mean pooling does. It is not evidence
that a model generalizes compositionally to unseen arrangements. Independent
calibration makes the latent-component association non-circular, while the
linearity itself is still an arithmetic identity. Subsampling can introduce
variation in the five bars; results are not smoothed or retuned.

## Outputs and reproducibility

Files are written under `OUTPUT_DIR` with checkpoint/image names:

- `*_composition.png`: the complete figure.
- `*_composition.npz`: full mu_A, mu_B, region means, mixture means, accepted
  patch probabilities and indices, and object-purity fractions.
- `*_composition.json`: source paths, checkpoint metadata, object colors,
  reference regions and accepted indices, preprocessing geometry, seed,
  normalization/temperature/overrides, selected prototype IDs, patch counts,
  mixture indices, and all component masses.

Changing experiment settings and rerunning the same checkpoint/image overwrites
these files. Use a distinct `OUTPUT_DIR` for configurations you want to retain.
