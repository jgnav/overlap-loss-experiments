# Region aggregation ablations

Set `region_aggregation` in `config/train.yaml`. `mean` is the unchanged
baseline. `region_patch_threshold` still selects patches first: a numerical
threshold gives equal weight to accepted patches; `weighted` assigns each
intersecting patch its overlap-area fraction. We normalize those weights to
sum to one within each view/region for all statistics and empirical CDFs.

| Value | Symmetric cross-view region loss |
| --- | --- |
| `mean` | Cross-entropy of arithmetic probability means |
| `mean_scalar_variance` | Mean loss + squared difference of total standard deviations |
| `mean_variance` | Mean loss + average squared difference of per-dimension standard deviations |
| `mean_covariance` | Mean loss + squared Frobenius covariance difference |
| `mean_projected_variance` | Mean loss + average squared difference of projected standard deviations |
| `mean_projected_covariance` | Mean loss + squared Frobenius projected covariance difference |
| `swd` | Sliced Wasserstein squared distance alone |
| `mean_centered_swd` | Mean loss + SWD of centered patch distributions |
| `mean_normalized_swd` | Mean loss + projected scale matching + standardized residual SWD |

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

The global DINO and iBOT/iBOT++ patch objectives are unaffected. Empty region
pairs remain excluded; distributed losses retain global valid-region
normalization, including empty ranks. Changing aggregation when resuming an
existing run is rejected; use a new continuation to start a new ablation.
