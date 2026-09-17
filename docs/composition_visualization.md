# Controlled A + B composition figure

`composition_visualization.py` produces one 300-DPI PNG from a full iBOT
checkpoint, a numeric COCO image ID, and your datasets root. Lookup is restricted
to the COCO directory and its mask is generated from local instance annotations.
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
There is no same-image fallback. At least two reference regions per concept are
required so the pure controls can be evaluated leave-one-reference-out.
Additional regions improve the fingerprint estimates. Reference A and B must have the intended semantic meaning;
mask colors alone cannot establish that meaning.

The root must contain `coco/annotations/instances_{train,val}2017.json` and
images in either `coco/images/{train,val}2017` (the `prepare_data.py` layout) or
`coco/{train,val}2017` (the official archive layout). `DATASETS_ROOT` may also
point directly to that `coco` directory. No precomputed PNG mask is needed.
Install `pycocotools` in the active environment for faithful
polygon and compressed/uncompressed RLE decoding. The script checks only the two
standard COCO instance files and then constructs the image path from the matched
split and recorded filename. It never traverses ImageNet, VOC, ADE20K, or other
datasets. Missing or ambiguous matches fail with an actionable error.

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
projection head and supports the repository's shared/separate head checkpoints.
Centering is applied only when that probability mode is selected. `PATCH_PURITY = 0.9` selects patches
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
  The independently observed A+B crop forms a separate bank. The source image's
  segmentation-selected A/B patches form a third bank used only for the sanity
  panel. Assignments are not refit for each sanity mixture, since balancing each
  pure region could erase the very contrast being measured. Sinkhorn results
  depend on this recorded assignment scope and should not be interpreted as
  invariant to a change in batch composition.
- `raw_logits` checkpoints are rejected because signed L2-normalized vectors
  do not define probability mass. An explicit `NORMALIZATION_OVERRIDE = "softmax"`
  is allowed, but then the figure visualizes that override, not the checkpoint's
  trained raw-vector region representation. The override is recorded in JSON.

Pure-region reference means are averaged **equally across regions** to obtain
the full-dimensional fingerprints mu_A and mu_B. No prototype dimension is
discarded from the measurement. For each fingerprint the script reports top-K
probability mass for `CONCENTRATION_K` (default 8, 32, 128, 512) and the
entropy-effective prototype count `exp(H(mu))`.

The mixed representation is obtained from a new model forward over a real crop
enclosing both selected COCO instances. All sufficiently complete image patches
in this crop are averaged; the representation is not assembled from A- and
B-segmentation patches. The script then solves nonnegative least squares over
every prototype dimension:

`r_AB ~= alpha*mu_A + beta*mu_B + epsilon`.

The stacked decomposition displays normalized L1 magnitudes
`[alpha, beta, ||epsilon||_1]`. JSON also records the unnormalized coefficients,
residual norm, and cosine similarity between the observation and reconstruction.
Pure A/B controls are averages of leave-one-reference-out fits, so a pure region
is never fitted using a fingerprint containing itself.

The PNG contains:

1. Source image with object-mask boundaries.
2. The independently observed A+B crop and the crop patches used in its mean.
3. Top-K mass and entropy-effective support diagnostics for mu_A and mu_B.
4. The most discriminative prototype dimensions by `|mu_A-mu_B|`, using one
   common probability scale. This heatmap is illustrative only.
5. Full-dimensional NNLS decompositions for leave-one-out pure controls and the
   observed A+B crop.
6. An explicitly labeled sanity check with five constructed patch-count mixtures.
7. The composition-test formula and protocol summary.

All dimensions remain intact during concentration and decomposition computation;
only the heatmap selects dimensions for display. No PCA, UMAP, clustering, or
learned projection is performed.

## Controlled mixtures and interpretation

The mixture count is the largest multiple of four no greater than
`min(MIXTURE_PATCHES, accepted_A_count, accepted_B_count)`. At least four pure
patches per object are required; the script never silently lowers purity or
samples patches with replacement. A fixed seed establishes one patch ordering
per concept. Mixtures use prefixes of those orderings, keeping total patch count
constant and achieving the requested ratios exactly. A-only and B-only mixture
endpoints may therefore differ from the all-patch region means above them.

Panel f uses controlled **patch subsets in a fixed image**, not physically pasted
images or new model forwards. Its all-patch union obeys
`r_AB = (n_A*r_A + n_B*r_B)/(n_A+n_B)` by construction; the maximum absolute
residual is recorded. This illustrates what mean pooling does. It is not evidence
that a model generalizes compositionally to unseen arrangements. It is retained
only as an implementation sanity check. The primary evidence is the separately
forwarded observed A+B crop in panels b, d, and e.

## Outputs and reproducibility

Files are written under `OUTPUT_DIR` with checkpoint/image names:

- `*_composition.png`: the complete figure.
- `*_composition.npz`: full mu_A, mu_B, per-reference means, the observed mixed
  representation, full fit outputs, mixed-crop probabilities, sanity-mixture
  means, selected indices, and object-purity fractions.
- `*_composition.json`: source paths, checkpoint metadata, object colors,
  reference regions and accepted indices, preprocessing geometry, seed,
  normalization/temperature/overrides, heatmap prototype IDs, concentration
  diagnostics, full-dimensional fit coefficients/errors, crop geometry, and
  sanity-mixture indices.

Changing experiment settings and rerunning the same checkpoint/image overwrites
these files. Use a distinct `OUTPUT_DIR` for configurations you want to retain.
