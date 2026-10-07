# Region aggregation ablations

Set `region_aggregation` in `config/train.yaml`. `mean` is the unchanged
baseline. `region_patch_threshold` still selects patches first: a numerical
threshold gives equal weight to accepted patches; `weighted` assigns each
intersecting patch its overlap-area fraction. We normalize those weights to
sum to one within each view/region for all statistics and empirical CDFs.

| Value | Symmetric cross-view region loss |
| --- | --- |
| `mean` | Cross-entropy of arithmetic probability means |
| `region_token` | Cross-entropy of learned tokens that attend only to selected overlap patches |
| `hellinger` | Cross-entropy of normalized squared means of square-root probabilities (fixed r=0.5) |
| `mean_scalar_variance` | Mean loss + squared difference of total standard deviations |
| `mean_variance` | Mean loss + average squared difference of per-dimension standard deviations |
| `mean_covariance` | Mean loss + squared Frobenius covariance difference |
| `mean_projected_variance` | Mean loss + average squared difference of projected standard deviations |
| `mean_projected_covariance` | Mean loss + squared Frobenius projected covariance difference |
| `swd` | Sliced Wasserstein squared distance alone |
| `mean_centered_swd` | Mean loss + SWD of centered patch distributions |
| `mean_normalized_swd` | Mean loss + projected scale matching + standardized residual SWD |

`hellinger` computes `q = normalize((sum_i w_i sqrt(p_i))^2)` independently
for teacher and student, followed by the existing symmetric cross-view
cross-entropy. Weights are selected before pooling as above. It adds no
auxiliary losses or tunable exponent. Student pooling uses log space for
stable gradients. Centered teacher probabilities reuse the ordinary iBOT
targets. This option supports `centering`, `softmax`, and `sinkhorn`;
`raw_logits` is rejected because signed vectors have no real square-root
probability embedding. To try it, set `region_aggregation: hellinger` in a training YAML.

All moments use population normalization (no Bessel correction). Variance
variants match `sqrt(variance + 1e-8)`. Fixed settings live in
`losses/region_aggregation.py`: all auxiliary coefficients are 1, 64 unit
Gaussian projection directions, projection seed 0, and 128 shared midpoint
quantiles. Unit coefficients are a neutral starting point, not a claim of
optimal relative scaling across losses. `lambda3` weights the complete region
objective as before. There are no auxiliary-weight YAML options.

Directions are fixed across steps, views, teacher/student, ranks, and restarts.
They are reconstructed using a local CPU generator, without consuming the
training RNG. SWD uses inverse weighted empirical CDFs at the same quantiles,
so different patch counts and area weights are supported. This is a finite
quantile approximation to 1-D W2 squared, averaged over directions. Normalized
SWD detaches the standard deviation in its shape denominator, as requested.
Full covariance uses exact Gram identities instead of a D-by-D allocation.

Teacher targets are detached. The same selected patch distributions are used
for moments under softmax, centering, and Sinkhorn. With `raw_logits`, the
extensions operate on signed L2-normalized patch vectors and retain cosine
distance for the mean term. They are vector-distribution ablations in that case.
In Sinkhorn mode, teacher moments use detached overlap-patch assignments and
student moments use ordinary softmax at `student_temp`, as does the mean term.

The global DINO and iBOT/iBOT++ patch objectives are unaffected. Empty region
pairs remain excluded; distributed losses retain global valid-region
normalization, including empty ranks. Changing aggregation when resuming an
existing run is rejected; use a new continuation to start a new ablation.

## Learned overlap token

Select `region_aggregation: region_token`, or use
`config/ablations/region_aggregation_region_token.yaml`. This config differs
from `config/train_ablation.yaml` only in the aggregation value.

Following CRISP Sec. 3.3 / Eq. (5), one learned token is prepended to each
view's final backbone patch embeddings in a separate one-block aggregation
module. Only its attention query is evaluated: its keys/values are the selected
overlap patches of that image. It cannot attend to itself, CLS, registers, or
excluded patches. The token gets an attention residual, an MLP residual, and
final LayerNorm. Patch queries are unnecessary for a single block; omitting
them also avoids the all-masked query rows in the paper's literal mask.
The backbone retains its ordinary image context, as in CRISP; the restriction
applies to the aggregation block, not earlier backbone attention.

The same geometry, minimum area, threshold, flips, and empty-pair exclusion as
the mean baseline are used. With `weighted` selection, coverage fractions are
attention priors (additive log weights). No extra minimum patch count is added.
Student tokens read the ordinary masked student features; teacher tokens read
unmasked teacher features. The region embedding uses the existing patch
projection/prototype head, preserving the configured head sharing. There is
no separate projection head or change to the DINO/iBOT losses.

The region loss is half the sum of the two opposite-view cross-entropies.
Student temperature stays `student_temp`; teacher temperature is `region_temp`.
With default `centering`, a separate region-logit center starts at zero and is
updated from both views of valid pairs across ranks with `center_momentum2`.
Ordinary iBOT centers remain independent. `softmax` omits this center;
`sinkhorn` assigns only valid teacher region tokens across views/ranks, while
the student uses ordinary softmax. Raw-logit and deep supervision combinations
are rejected. This variation supports `region_views: global` and
`loss_modality: standard`; gradient accumulation is currently rejected.

The new token/block train with the existing optimizer and schedules; the
teacher copies their initialization and follows the existing EMA. Continuation
from ordinary iBOT accepts these new parameters. Checkpoints save the student,
teacher, optimizer, and independent region center for full resume. The default
ablation keeps 50 epochs and `lambda3: 0.4`.
