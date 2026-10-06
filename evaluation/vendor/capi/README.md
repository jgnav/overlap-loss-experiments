# Pinned CAPI evaluation code

Source: https://github.com/facebookresearch/capi/blob/98b4fa17ee8eec8810c17022df9a27a44845368b/eval_segmentation.py

Revision: `98b4fa17ee8eec8810c17022df9a27a44845368b` (Apache-2.0; license included).

`eval_segmentation.py` copies upstream metrics, classifiers, hyperparameter
selection, and `eval_model`. Local integration changes are limited to:

- Import data/extraction/logging adapters from `evaluation.utils.capi_adapter`.
- Postpone type annotations and omit annotation-only jaxtyping imports.
- Import cuML inside the logistic-regression fit rather than at module import.
- Omit upstream's OmegaConf CLI (`main`); retain our existing CLI/JSON output.
- Break equal-score parameter ties by original grid index, preserving the
  single-rank choice when the search is distributed across ranks.

The classifier calculations and evaluation flow are unchanged. The adapter
supplies pre-downloaded datasets (VOC `train`, not `trainaug`), final normalized
backbone patch tokens, row-major patch pixel labels, and plain progress prints.
The adapter shards extraction across all visible GPUs and restores original
image order on every rank using bounded tensor transfers. CAPI distributes
parameter candidates across ranks; its final refit/scoring remains on rank 0.
CAPI still performs its own NumPy
10% holdout, feature standardization, sweep, refit, and final scoring.

Resolution retains the user-selected CRISP adjustment: 256 pixels for /16
and 224 for /14, giving 256 patches in both cases. This differs from upstream
CAPI's fixed 224-pixel segmentation input. Labels and shard gathering use the actual patch grid.
ADE20K k-NN uses the bfloat16 override from the released default evaluation YAML.
VOC uses the user-selected original train/val splits; published VOC-score
reproduction is not guaranteed by upstream's loader either.

## Classification

`eval_classification.py` is from the same revision; `classification_support.py`
copies upstream InfiniteSampler, make_data_loader, DatasetWithEnumeratedTargets,
MetricLogger and SmoothedValue. Local changes:

- Deferred annotation-only imports and removed upstream CLI/OmegaConf dependencies.
- Local ImageFolder/checkpoint adapter; independent copies preserve training and
  holdout transforms. The adapter matches CAPI's released iBOT model-output API.
- Persist final holdout sweep and selected test classifiers for JSON reporting.
- Atomic checkpoint/symlink replacement and retain only the latest checkpoint.
- Fail explicitly on nonfinite loss; do not change classifier calculations.
- Convert local ImageFolder identifiers to path strings in result keys.

The vendor core retains upstream implementations. The local adapter overrides
the training budget with CRISP's 200 epochs of image exposure and replaces the
LR grid with base LR 0.001 (actual peak 0.004 at global batch 1,024).
AdamW, 1,250-step warmup, three weight-decay candidates per feature, initializers,
attention head, multi-head loss, seed-42 split/sampler, padding and selection
remain upstream. These overrides are recorded as CRISP settings with CAPI
fallback assumptions; this is not the unchanged CAPI protocol. Batch per GPU is 1024 / GPU count (256 on four).
Eager execution (`use_compile=False`) avoids compilation of the 12-head graph;
it leaves the mathematical protocol unchanged. No mixed precision is introduced.

Source: https://github.com/facebookresearch/capi/blob/98b4fa17ee8eec8810c17022df9a27a44845368b/eval_classification.py
