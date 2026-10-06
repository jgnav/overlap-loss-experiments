# Offline video object segmentation

## Default protocol

The offline evaluator selects `video_protocol: dinov3`, `video_resolution: small`,
and `video_feature_blocks: 4`. ViT-S/16 inputs use a short side of 480 and the
long side rounded to the nearest multiple of 16. A native 854 × 480 frame becomes
848 × 480. RGB uses PIL bicubic interpolation and ImageNet normalization
(mean .485/.456/.406; std .229/.224/.225).

The frozen teacher backbone returns the last four LayerNorm-normalized patch
blocks. Their mean is L2 normalized before matching. Averaging four blocks is the
user-requested feature modification; the DINOv3 paper uses the last block.
`video_feature_blocks: 1` remains available for that comparison.

Propagation uses the published DAVIS-training-selected settings: first-frame
labels plus seven preceding soft predictions, unrestricted spatial matching,
top5 including ties, and cosine-softmax temperature0.2. Only objects present in
the first labeled frame are initialized and scored; later annotations never introduce
new labels. YouTube-VOS archive frames before the first provided annotation are
skipped to align with the official labeled clip start. Soft mask probabilities are bilinearly resized directly to native
resolution, normalized per channel as in the released notebook, and converted
to indexed labels. Metrics are native-resolution J, F, and their mean, averaged
first over annotated frames per object and then over objects. DAVIS endpoints
are excluded; other datasets score available annotated frames after frame0.

The paper specifies nearest patch-multiple resizing and bilinear native mask
upsampling. The released demonstration notebook instead rounds upward and uses
nearest mask upsampling. We follow the paper for these explicit choices and
retain notebook details for unspecified operations. The full author benchmark
harness and custom split lists are not released.

## Sources

- [DINOv3 paper, Section6.1.5, AppendixD.5 and Table27](https://arxiv.org/html/2508.10104v1#A4.SS5).
- [Released tracking notebook, pinned commit](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/notebooks/segmentation_tracking.ipynb).
- Downloaded paper: `output/papers/dinov3_2508.10104v1.pdf`.

## Explicit available-data evaluation splits

| Dataset | Evaluation set | RGB and scoring |
| --- | --- | --- |
| DAVIS2017 | Official validation, 30 videos | All frames; standard native J/F |
| YouTube-VOS2019 | Official validation, 507 videos | All RGB frames from the released ZIP; score supplied annotations only |
| MOSEv2 | Seed0 holdout of 301 annotated training videos | All released RGB frames and masks |

The two non-DAVIS splits are installed-data adaptations. They do not claim to
match DINOv3's custom 690-video YouTube-VOS or original-MOSE 301-video test lists.
The manifests explicitly record versions, split provenance and
`author_split_verified: false`. YouTube-VOS uses first-frame objects and an
all-object J/F average under this protocol, rather than the challenge server's
seen/unseen averaging and later-object initialization rules.

MOSEv2's remaining 3365 training videos and YouTube-VOS's 3471 training videos
are recorded as disjoint selection sets but are unused: propagation parameters
are fixed to the published DAVIS-selected values, without tuning on any test set.
MOSEv2 official validation has initial masks only and cannot produce local J/F;
the annotated training holdout supports offline scoring.

`config/video_splits/*.json` contains the exact IDs. Paths resolve relative to
`datasets_root`, keeping manifests usable on another installation. Rebuild with:

```bash
python evaluation/prepare_video_splits.py /path/to/datasets
```

The builder sorts IDs before seeded sampling. ZIP access validates frame names,
checks all available annotation/RGB alignment, and decompresses frames on demand.
ZIP CRCs, names and sizes form a recorded archive index hash. No full archive
extraction or extra download is needed.

## Run and resume

`config/evaluation_video_dinov3.yaml` enables only the three video tasks for the
200-epoch region ViT-S/16 teacher checkpoint. General offline configs use the
same video settings and manifests. Each video task uses one GPU; all other
probe/training protocols retain their existing configurations.

```bash
sbatch slurm/evaluation_video_dinov3.sh config/evaluation_video_dinov3.yaml
```

The Slurm launcher requests one GPU, four CPUs, 24GiB RAM and twelve hours.
Separate task configs allow independent scheduling. The launcher requeues near
its time limit. Atomic per-video progress is reused only when code, checkpoint,
feature selection, resolution, split manifest and archive identity match.
Historical output is preserved; launches use new output directories.

## Historical protocol comparisons

Completed DAVIS comparisons, all 30 videos and 61 first-frame objects:

| Protocol / features | Original iBOT J&F | Region200 J&F |
| --- | ---: | ---: |
| DINOv3 nearest rounding / last1 | 61.84467 | 59.96713 |
| DINOv3 nearest rounding / mean last4 | 63.30027 | 62.78777 |
| DINOv3 released upward rounding / mean last4 | 63.35936 | 63.12255 |
| Original DINO480p / mean last4 | 63.02537 | 62.44312 |
| Original DINO square480 / mean last4 | 53.91991 | 55.67554 |

Historical measurements remain under `output/analysis/`. Original DINO adapters
and the pinned source/license are retained for reproducing those comparisons;
`dino_480p_last4` and `dino_square_last4` require explicit selection. They are not
the default offline video protocol.
