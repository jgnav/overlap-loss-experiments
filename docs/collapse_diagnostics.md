# Collapse diagnostics

These are detached observations in the existing `train.py` path. They do not
change the loss, assignment normalization, optimizer, schedules, teacher EMA,
model forward computation, or checkpoint state. Settings live in the existing
`config/train.yaml`; defaults also apply to older configurations.

## Prototype geometry

The existing head topology is selectable in `config/train.yaml`. With
`shared_head: true`, CLS and patch tokens use the same final prototype layer;
with `shared_head: false`, they use separate CLS and patch prototype layers
constructed by the same weight-normalized implementation. The single setting
is applied to both the student and EMA teacher; the diagnostic reports a shared
layer only once as the patch layer.

Reference: [Why Prototypes Collapse, Definition 2.1](https://arxiv.org/html/2510.20108v2#S2).
The official [proto-decoupling repository](https://github.com/dsb-ifi/proto-decoupling)
was inspected at commit `d40f4f092640b1428c4e3adea53372ee0caee268` (2026-09-14),
including its recursive file listing, Python files, and README. It exposes
training/GMM code but no explicit reusable unique-prototype counting helper.
No runtime dependency on that repository is introduced.

Our fallback is a deterministic representative cover, not an exact reproduction
of the paper's evaluation implementation and not a minimum-cardinality cover:

1. Read the effective final output weights and L2-normalize their rows.
2. For each epsilon independently, scan rows in original prototype-index order.
3. The first uncovered row becomes a representative. Assign every still-uncovered
   row with **strict** cosine distance `1 - dot < epsilon` to that representative.
4. Continue until every row is covered. Count representatives; divide by K for
   the unique-prototype ratio.

Every resulting partition contains its representative, and all its members
meet the distance criterion to that representative. This satisfies Definition
2.1. It is order-dependent, not connected-component clustering: chains of nearby
vectors cannot merge arbitrarily distant endpoints. Different index orders can
produce different valid covers. Independent threshold covers are not guaranteed
to be nested. All comparisons use CPU FP64; zero-length/nonfinite vectors raise
an explicit error rather than fabricate a normalized prototype count.

`iBOTHead.prototype_layers()` follows the head's forward implementation:
`last_layer2` is the patch output, or `mlp2` without a bottleneck. A shared layer
is reported only as `patch`; distinct CLS layers receive additional `cls`
metrics. Student and teacher are always measured separately. Legacy weight
normalization's cached `.weight` can be stale after an optimizer/EMA update;
the diagnostic recomputes effective weight from current g/v through the
weight-normalization hook's `compute_weight`, without updating the cached
attribute or executing a forward pass. It never counts internal `weight_v`
directions directly. A modern parametrized layer uses its effective `.weight`.

Prototype geometry runs **once after each completed epoch**, on rank zero, after
the last optimizer/EMA updates. DDP-synchronized heads need no weight gathering;
only the scalar metrics are broadcast. CPU similarity chunks have at most
`diagnostic_prototype_chunk_size x K` FP64 entries (default 256; 16 MiB for K=8192).
There is no persistent K x K tensor and no GPU similarity matrix.

Metric suffixes are `0025`, `0100`, `0250`, `0500` for epsilon 0.025, 0.1, 0.25, 0.5.
For each threshold the training statistics contain:

- `student_patch_unique_prototypes_eps_<suffix>`
- `student_patch_unique_prototype_ratio_eps_<suffix>`
- `teacher_patch_unique_prototypes_eps_<suffix>`
- `teacher_patch_unique_prototype_ratio_eps_<suffix>`

## Backbone representations

The existing `return_backbone_feat=True` multicrop path returns the same teacher
outputs plus its final normalized backbone tokens, before the projection head.
CLS is excluded. Only the teacher's unmasked global crops participate; no extra
forward or ImageNet pass is run.

Defaults: the first **one** training batch of each epoch, at most **4096 patches
per rank per sampled batch**. For P available patch positions in flattened
view/image/patch order and L=min(P, cap), select positions `floor(j * P / L)`
for j=0,...,L-1. This includes both equally sized global views, uses no RNG,
and applies the same policy in every loss condition. Increase
`diagnostic_feature_batches` or `diagnostic_max_patch_features_per_batch` to
change this sample. Metrics describe that global sample, not the whole epoch.

Sampled features are detached and accumulated on CPU in FP64: count, sum of
features, sum of outer products, and sum of individually unit-normalized
features. At epoch end, all ranks all-reduce these sufficient statistics before
constructing covariance and computing eigenvalues. NCCL briefly stages just the
packed statistics on GPU (about 1.13 MiB for D=384); eigendecomposition is CPU.
Per-rank effective ranks, spectra, or mean-direction norms are never averaged.

Covariance uses denominator N. Negative roundoff eigenvalues are clamped to
zero. Spectral entropy gives effective rank, its ratio to D, and top-eigenvalue
variance fraction. Mean direction norm is computed without centering the
features. Zero feature vectors contribute zero direction. No-sample/zero-variance
cases report rank/ratio/top1 fraction as zero. A positive variance at or below
`10 * D * machine_epsilon_float64 * max(abs(second_moment))` is treated as a
subtraction-roundoff residual. Mean direction norm is still computed independently
(identical nonzero features give rank zero and mean direction norm one).

Metrics: `teacher_patch_feature_effective_rank`,
`teacher_patch_feature_effective_rank_ratio`,
`teacher_patch_feature_top1_variance_fraction`,
`teacher_patch_feature_mean_direction_norm`. An additional
`teacher_patch_feature_sample_count` records the actual global sample size.

## Logging and interpretation

All metrics join the ordinary epoch statistics: W&B uses `train/<metric>`, JSON
uses `train_<metric>`, and TensorBoard retains the existing bare metric tags.
Nothing is averaged over optimizer steps for these epoch diagnostics.

`student_patch_effective_prototypes` and `teacher_patch_effective_prototypes`
are removed. Patch entropy and maximum probability remain unchanged and measure
**assignment sharpness**, not prototype-vector collapse. Geometric uniqueness,
backbone effective rank, and mean direction concentration answer separate
questions; none is relabeled as an entropy-based assignment count.
