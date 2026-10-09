# Probe3D dataset readers

`evaluate_spair_correspondence.py` and `configs/spair_correspondence.yaml` are
also copied verbatim from the same upstream commit. `evaluation/spair_neco.py`
uses the released `compute_errors` matcher for NeCo's stated 224-pixel SPair
evaluation, and reports the paper-caption threshold 0.01 separately from the
upstream default 0.1.

`evals/datasets/{spair,navi,scannet_pairs,utils}.py` are vendored from
[mbanani/probe3d](https://github.com/mbanani/probe3d), commit
`a1f14640076e38b8bc07b66d0fe2d01d15691e9d`, under the adjacent MIT
license. `spair.py` also carries its upstream HPF Apache 2.0 notice; the
adjacent `LICENSE-APACHE-2.0` covers that code. They retain the released split, pair-selection, image preprocessing,
and geometry conventions. `evaluation/utils/correspondence.py` supplies an
iBOT checkpoint adapter and a chunked PyTorch implementation of Probe3D's
cosine 2-NN ratio ranking, avoiding a separate FAISS-GPU installation.
