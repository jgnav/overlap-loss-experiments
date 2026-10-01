# Regional ordering performance

The optimized implementation preserves the loss definitions, reference input
order, sampling step, selected patches, temperatures, coefficients, masks,
gradient flow and checkpoint state. Cross-image still uses one global/global
overlap per image and all eligible external references. Within-image still
uses all other distinct eligible regions in the query image.

The original [NeCo implementation](https://github.com/vpariza/NeCo/blob/main/src/neco.py)
uses batched cosine matrix multiplication before calling its sorter. Our
previous implementation instead gathered a separate `[queries, references,
8192]` prototype tensor. At 96 local cross-image queries and 191 references,
that temporary alone contains about 574 MiB of float32 values and was rebuilt
in backward. Within-image queries could produce many such tiles per step.

Changes:

- Share the reference bank in matrix multiplication. Cross-image gathers only
  scalar cosines; within-image shares one bank per image in batched multiplication.
- Pool all required images together per crop view, with softmax tiles limited
  to 64 MiB. Recompute probabilities in backward and allocate each view's
  logit gradient once. This avoids repeated batch-sized selection gradients.
- Group patch grids for binary geometry instead of computing unused fractional
  weights on a second patch grid for every crop/region comparison.
- Use cached partner-wire permutations in the bitonic sorter instead of repeated
  advanced indexing and copying full permutation matrices at every comparator stage.
- Reuse identical student/teacher sorting queries. Compute detached teacher
  permutations once; checkpoint the student sorter without rerunning the teacher.
- Keep sampler random state identical while avoiding redundant size-one shuffles.

No new packages, model parameters, YAML options, or checkpoint buffers are required.
The pooling operation provides first-order training gradients; higher-order
derivatives are not supported by its memory-efficient backward.

## CPU measurements

Single-thread synthetic loss-only forward/backward, 8,192 prototypes, two global
14-by-14 grids and ten local 6-by-6 grids. Cross-image uses a simulated bank of
four ranks; communication, backbone, optimizer, data loading and online probes
are excluded. Median of two measured steps after one warmup, against the
pre-optimization implementation at repository commit `8467ce4`:

| Mode | Local batch | Queries | References/query | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-image | 48 | 96 | 191 | 15.146 s | 2.385 s | 6.35x |
| Within-image | 8 | 832 | 46 | 2.675 s | 1.101 s | 2.43x |

The benchmark compares loss values and all student-logit gradients as well as
timings. The gains depend on shape and hardware; these measurements do not
predict GPU epoch times. Small CPU-only shapes can be dominated by batching
overhead. No GPU code was run during validation.

For repeatable CPU measurements, save the previous `region_ordering_loss.py`
and `region_sorting.py` together in a reference directory, then run:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python benchmarks/benchmark_region_ordering.py \
  --reference-source /path/to/reference --batch-size 48 --prototypes 8192 \
  --modality cross_image --iterations 2

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python benchmarks/benchmark_region_ordering.py \
  --reference-source /path/to/reference --batch-size 8 --prototypes 8192 \
  --modality within_image --iterations 2
```

Omit `--reference-source` to measure just the current implementation. Add
`--profile` for CPU operator counts. The script never launches a GPU or backbone.

## Patch-wise regional rank distribution

The third modality, `patch_rank_distribution`, retains the optimized geometry
and bitonic wiring. Its continuous backbone feature bank uses shared matrix
multiplication, gathers only scalar similarities and samples at most 49 external
regions. Patch permutation matrices are reduced into regional means in tiles
of at most 128 patches (also bounded by the permutation-element budget). Only
scalar cosine vectors and patch-to-region indices are retained for student
backward; the sorter is recomputed one tile at a time. Teacher sorting is
performed once without gradients. This gives exact first-order gradients for
the mean-of-matrices objective without retaining every patch's sorting stages.

A single-thread CPU benchmark at batch size 48, feature dimension 384, native
14x14 grids, two global views, four simulated reference-bank ranks and 49
references measured **5.763 seconds** for one forward/backward step after a
warmup, with finite gradients. This is a loss-only CPU measurement, excludes
communication/backbone/optimizer, and does not predict GPU epoch time. Sorting
every patch is inherently more work than sorting one mean per region.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python benchmarks/benchmark_patch_rank_distribution.py \
  --batch-size 48 --features 384 --iterations 1
```
