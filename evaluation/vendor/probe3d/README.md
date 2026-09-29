# Probe3D dataset readers

`evals/datasets/{spair,navi,scannet_pairs,utils}.py` are vendored from
[mbanani/probe3d](https://github.com/mbanani/probe3d), commit
`a1f14640076e38b8bc07b66d0fe2d01d15691e9d`, under the adjacent MIT
license. `spair.py` also carries its upstream HPF Apache 2.0 notice; the
adjacent `LICENSE-APACHE-2.0` covers that code. They retain the released split, pair-selection, image preprocessing,
and geometry conventions. `evaluation/utils/correspondence.py` supplies an
iBOT checkpoint adapter and a chunked PyTorch implementation of Probe3D's
cosine 2-NN ratio ranking, avoiding a separate FAISS-GPU installation.
