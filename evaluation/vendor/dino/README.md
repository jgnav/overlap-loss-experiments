# Original DINO video evaluator

Unmodified upstream `eval_video_segmentation.py`, pinned to the last commit
changing that file: `4b96393c4c877d127cff9f077468e4a1cc2b5e2d`.

Source: https://github.com/facebookresearch/dino/blob/4b96393c4c877d127cff9f077468e4a1cc2b5e2d/eval_video_segmentation.py

SHA256: `f17eb52954b89e3bc5cef99186fafe23826bd97810c9de6953715832b94b1d67`.
Apache 2.0 license copied alongside the source. This file is retained for audit
and numerical equivalence tests; it expects the original DINO repository for
standalone execution.

`evaluation/utils/video_dino.py` adapts it to this repository's checkpoint API
using original 480p resizing and averaging the last four LayerNorm-normalized
patch blocks. Historical square resizing remains an explicit alternative. Propagation is chunked with the same
exp/local-mask/top-k/sum equations. Predicted indexed masks are exported, and
native-resolution J/F scores use the vendored official DAVIS scorer.
