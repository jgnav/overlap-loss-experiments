# DAVIS metrics

`metrics.py` is vendored from
[davisvideochallenge/davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)
under the adjacent BSD 3-Clause license. It computes region Jaccard and
boundary F for each object mask. The video evaluator performs DINO-style
label propagation and aggregates these metrics into Jm, Fm and J&F.
