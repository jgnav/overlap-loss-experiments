# Ablation implementation audit — 30 September 2026

This review compares the current loss code, its Git history, ablation YAML
files, and the recorded W&B training arguments for the runs used in the table.
The initial repair restored Sinkhorn normalization. The subsequent requested
repair also fixes softmax student temperature and iBOT++ token weighting.
Documentation, regression tests, and checkpoint compatibility markers track
those changes. Optimizer behavior and training configurations are unchanged.

## Restored Sinkhorn behavior

`region_normalization: sinkhorn` now follows the earlier teacher-only
normalization:

1. Compute the configured overlap geometry and patch-selection weights.
2. Select raw teacher logits only from accepted pairs and selected overlap
   patches. Unselected patches and rejected pairs never enter Sinkhorn.
3. Run three detached Sinkhorn iterations jointly over the selected teacher
   patches from both global views and all distributed ranks.
4. Apply ordinary student softmax at `student_temp` (0.1 in the existing
   ablation recipe). Teacher Sinkhorn uses the existing `region_temp` (0.07).
5. Pool both sides with the configured weights and aggregation, then compute
   symmetric cross-view cross-entropy. Gradients flow only through the student
   softmax, pooling, and any student statistics.

The stable log-space Sinkhorn implementation is retained. The original
probability-space iteration and the log-space iteration implement the same
balancing for finite inputs, without the original exponential overflow risk.

Patch selection remains an independent ablation: numeric thresholds select
equal-weight patches; `weighted` selects every positive overlap fraction and
pools by area. Restoring normalization does not replace threshold 1.0 with the
old area-weighted geometry.

New Sinkhorn checkpoints record `region_loss.sinkhorn_teacher_only` in the
loss state. Exact resume rejects student-Sinkhorn checkpoints that lack that
marker when the region loss is active. The other resume checks and optimizer
loading behavior are unchanged. The saved run `83650_6` still measures the
previous objective; changing source code does not update its scores. A fresh
experiment is required to measure the restored normalization.

## When Sinkhorn changed

All times below are BST, as recorded in Git.

| Commit | Date and time | Change |
| --- | --- | --- |
| `93db8bf` | 2026-09-09 11:45:46 | Teacher Sinkhorn operated on raw selected overlap-patch logits before area-weighted pooling. The student used temperature-scaled softmax. |
| `8712ce1` | 2026-09-16 19:24:08 | Removed the teacher-only Sinkhorn path and replaced the region objective with uncentered softmax on both sides, with configurable binary patch selection. |
| `27362c8` | 2026-09-16 19:44:53 | Introduced `region_normalization: sinkhorn`, applying Sinkhorn independently to teacher and student and differentiating through student distributed normalization. |
| `dd60b5a` | 2026-09-17 16:20:31 | Added centered normalization using ordinary iBOT teacher targets and the ordinary student temperature. |

The completed Sinkhorn run `83650_6` used the student-and-teacher version.
Teacher-only overlap selection and student Sinkhorn are distinct questions:
the previous ablation already excluded unselected patches, but also balanced
student patches and differentiated through that balancing.

## Findings in other ablations

### iBOT++: unequal token weighting relative to the cited objective

Before the follow-up repair, `losses/ibot_loss.py` computed:

\[
L_{\mathrm{patch}} = \operatorname{mean}_{i\in M}\ell_i
                    + \operatorname{mean}_{i\in V}\ell_i.
\]

Thus each masked token receives weight `1 / |M|`, while each visible token
receives `1 / |V|`. The groups have equal total weight even when their sizes
differ, and the patch-loss scale also changes relative to ordinary iBOT.
For a masking fraction of 0.3, masked tokens get about 2.33 times the
per-token weight of visible tokens.

[TIPSv2 Section 3.3, Eq. (3)](https://arxiv.org/html/2604.12012v1#S3.SS3)
defines iBOT++ as a single sum over all patch tokens, with uniform token
weights. Averaging that sum over all tokens would give
`r * mean(masked CE) + (1-r) * mean(visible CE)`, where `r` is each sample's
masking fraction. A global loss coefficient cannot correct the unequal token
weights in the earlier implementation.

The saved run `83657_14` therefore measures an independently normalized
masked-plus-visible variant. It should be described as such, or rerun with the
paper's token weighting before being presented as that precise iBOT++
objective. The subsequent repair now computes one mean over all tokens;
masked and visible means are diagnostics only. New checkpoints carry
`ibot_plus_plus_all_tokens`, and exact resume rejects the old group-weighted
objective. The saved `83657_14` scores still measure the earlier variant.

### Softmax: student temperature changes alongside teacher centering

In the centered default, the student uses `student_temp=0.1` and the
teacher uses centered iBOT targets at `teacher_patch_temp=0.07`. In the plain
softmax branch before repair, both sides used `region_temp=0.07`.

The centered-versus-plain-softmax comparison therefore changes student
temperature as well as teacher normalization. This behavior is documented
and was introduced with the symmetric region-normalization design; it is a
confound when the table is interpreted as isolating teacher normalization.
The plain-softmax score `83650_7` is valid for its implemented objective, but
does not isolate removing teacher centering while holding student softmax
fixed. The subsequent repair now uses `student_temp` for student probabilities
in both the mean and moment terms, while retaining `region_temp` for the
uncentered teacher. New checkpoints carry
`region_loss.softmax_ordinary_student_temperature` and cannot silently resume
the former objective. The saved `83650_7` scores retain the earlier behavior.

### Historical deep supervision: mismatched input representations

Commit `c711ae0` (2026-09-25 12:25:33) implemented `softmax_deep` by applying
the iBOT projection/prototype head to intermediate backbone features, and
using final patch-head logits at depth 12. The `raw_logits_deep` branch
matched backbone features directly. These were not two normalizations of the
same head-free intermediate features.

Commit `647f86d` (2026-09-28 12:05:35) changed deep softmax to operate directly
on backbone features. Commit `23c8dba` (2026-09-28 13:07:46) subsequently
removed both deep modes. Existing historical deep scores, including runs
`80205_14`, `80205_17`, `83650_5`, and `83650_8`, must be attributed to the
implementation actually used; they cannot be presented as measurements of
the corrected head-free comparison. Neither deep mode is in the current six
table panels, although the supplied subsection text still mentions depth.

### Projection-head row: name does not identify the actual experiment

`shared_head_false.yaml` and run `83657_23` test separate CLS and patch
projection/prototype heads. Both are initialized by copying the pretrained
shared head. Region consistency continues to use the patch head.

The row should be labeled "Separate CLS/patch heads" or equivalent. A
dedicated region projection head would be a different experiment.

### Aggregation variants: formulas match the documented definitions

Mean pooling, weighted population moments, the Gram-matrix covariance
identity, and Hellinger pooling match their documented mathematical forms.
Teacher values are detached and the geometry weights precede the statistics.

The variance variants compare standard deviations, via
`sqrt(variance + 1e-8)`, rather than comparing variances directly. The
per-dimension standard-deviation penalty is averaged over all 8,192
prototypes. For probability vectors and unit auxiliary weight, its value is
bounded by `2 / 8192`, approximately 0.000244. This is a scale concern for
interpreting the experiment, not evidence of an incorrect covariance or
variance calculation. A small loss value alone does not establish a small
gradient; separate auxiliary-loss and gradient measurements would be needed
to assess its influence. The aggregation code was not changed.

### Remaining current table families

No additional implementation mismatch was found in the reviewed area filter,
patch thresholds/area weighting, region coefficient, registers, or KoLeo:

- The minimum shared area is measured in normalized original-image area.
- Thresholds measure each patch's fraction covered by the intersection.
- Weighted selection includes positive overlap and normalizes pooling weights
  within each region.
- `lambda3` scales the complete region loss and zero bypasses that branch.
- Registers are excluded from patch-head inputs and spatial losses. New
  teacher registers copy the student's initialization.
- KoLeo uses pre-head student CLS features, separate global-crop batches,
  local nearest neighbors, float32 distances, and the documented coefficient.

This is a code and regression-test review; it does not prove that any
particular setting improves representation quality.

## Configuration and result provenance

Every active ablation YAML differs from `lambda3_0p4.yaml` in one training
setting. The covariance configuration additionally lowers only the evaluation
batch size, from 128 to 32.

Recorded W&B training arguments for the completed table runs agree on the
reviewed pretrained source, architecture, source epoch, seed, 50-epoch
schedule, effective batch size, precision, temperatures, optimizer schedule,
weight decay, and momentum, except for each named ablation setting. The
older `70768_0` run supplying the lambda=0.1 row also matches those recorded
settings apart from lambda. YAML equivalence alone does not guarantee
identical code revisions.

The earlier fixed online evaluator was replaced by the full offline protocols
in commit `1f9a96b` on 2026-09-20. Comparisons to the former 400-train/
200-validation VOC cosine k-NN probe require reevaluation under the same
protocol. This review did not recover older completed Sinkhorn checkpoints
that are absent from the current output tree.

Two epoch-50 aggregate probe files, for `83650_6` and `83657_15`, flag a
compatibility error despite completed individual results with the matching
checkpoint fingerprint, dataset manifests, and declared protocol. Their
evaluator source hashes changed during the probes. The table used the
completed individual metrics, as previously disclosed. Matching metadata
does not by itself prove every code revision is behaviorally identical.

## Verification

The focused suite covered region normalization, weighted ablations,
aggregation, iBOT, KoLeo, head initialization, registers, training config, and
checkpoint loading: 71 tests passed and the distributed test initially failed
because the sandbox blocked Gloo's local connection. That remaining test
passed when rerun outside the sandbox on CPU with the loopback interface.
All 72 checks therefore passed.

The Sinkhorn regressions check the legacy teacher-only arithmetic reference,
student gradients, selection exclusions, extreme-logit stability, independent
student sample gradients, detached teachers, and two-rank behavior with
unequal patch counts, empty ranks, and globally empty batches. They also
verify that exact resume rejects the previous student-Sinkhorn semantics and
accepts a checkpoint carrying the restored teacher-only marker.

## Follow-up repairs and fresh runs

The corrected softmax branch keeps ordinary student temperature in its mean
and moment terms. The corrected iBOT++ branch averages CE uniformly over
all tokens, while preserving masked/visible diagnostic means. Checkpoint
markers prevent resuming the superseded objectives without an explicit new
continuation. YAML files, optimizer behavior, batch size, and schedules were
not changed.

All 87 focused tests passed after the follow-up repair, including both
two-process Gloo tests. Shell syntax and configuration preflight checks passed.

Slurm array `87146` was submitted on 2026-09-30 at 13:54 BST using
`sbatch --parsable --job-name=ablation-loss-fixed --array=5,6,12 slurm/slurm_ablation.sh`.

| Array task | Configuration | Output directory |
| --- | --- | --- |
| `87146_5` | `region_normalization_sinkhorn.yaml` | `output/ablation/87146_5` |
| `87146_6` | `region_normalization_softmax.yaml` | `output/ablation/87146_6` |
| `87146_12` | `ibot_plus_plus_true.yaml` | `output/ablation/87146_12` |

Each run starts from `checkpoints/ibot_vit_small.pth`, with 50 epochs, 4 GPUs,
batch size 64 per GPU, and the existing optimizer configuration. They have
fresh output directories and retain the old results for comparison.
Performance of the repaired losses remains unmeasured at submission.
