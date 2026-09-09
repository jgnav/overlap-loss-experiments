# Pinned CAPI segmentation evaluator

Source: https://github.com/facebookresearch/capi/blob/98b4fa17ee8eec8810c17022df9a27a44845368b/eval_segmentation.py

Revision: `98b4fa17ee8eec8810c17022df9a27a44845368b` (Apache-2.0; license included).

`eval_segmentation.py` copies upstream metrics, classifiers, hyperparameter
selection, and `eval_model`. Local integration changes are limited to:

- Import data/extraction/logging adapters from `evaluation.utils.capi_adapter`.
- Postpone type annotations and omit annotation-only jaxtyping imports.
- Import cuML inside the logistic-regression fit rather than at module import.
- Omit upstream's OmegaConf CLI (`main`); retain our existing CLI/JSON output.

The classifier calculations and evaluation flow are unchanged. The adapter
supplies pre-downloaded datasets (VOC `train`, not `trainaug`), final normalized
backbone patch tokens, row-major patch pixel labels, and plain progress prints.
Only one GPU is supported by this adapter. CAPI still performs its own NumPy
10% holdout, feature standardization, sweep, refit, and final scoring.

Resolution is passed explicitly as `16 * patch_size`: 224 for patch size 14,
256 for patch size 16. This is CRISP's token-count convention, not CAPI's default
224 for all models. No legacy feature cache is consumed by this evaluator.

This pins the released implementation, not a claim that CRISP used this exact
revision, VOC split, or dataset ordering. CAPI's warning about its VOC paper
results remains relevant to published-score comparisons.
