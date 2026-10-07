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
| MOSE (2023) | Seed0 split: 1206 validation / 301 test videos, evaluate test | All released RGB frames and masks |

YouTube-VOS remains an installed-data adaptation. MOSE now uses the original
2023 release's 1507 fully annotated training videos, matching DINOv3's dataset
version and 1206/301 split sizes. The author video lists and seed are unavailable;
our fallback sorts the original IDs, shuffles with Python `random.Random(0)`,
and reserves the first 301 for testing and the other 1206 for validation.
These manifests do not claim to match DINOv3's custom 690-video YouTube-VOS
or original-MOSE 301-video test IDs.
The manifests explicitly record versions, split provenance and
`author_split_verified: false`. YouTube-VOS uses first-frame objects and an
all-object J/F average under this protocol, rather than the challenge server's
seen/unseen averaging and later-object initialization rules.

MOSE's 1206 validation videos and YouTube-VOS's 3471 training videos
are recorded as disjoint selection sets but are unused: propagation parameters
are fixed to the published DAVIS-selected values, without tuning on any test set.
The original MOSE release was downloaded from the authors' linked
[Hugging Face repository](https://huggingface.co/datasets/FudanCVL/MOSE), with
the outer archive and inner training archive SHA256 checks verified. It is stored
under `datasets_root/MOSE2023`; `preparation_report.json` records provenance,
hashes and annotation alignment across the complete original archive. The 301
test videos are extracted for scoring; the unused 1206 validation videos remain
available in the verified archive. The split builder uses the original
`meta_train.json` population and requires every selected test video to be
extracted. Official MOSE validation has initial masks only
and cannot produce local J/F. The earlier MOSEv2 manifest is retained only for
reproducing the previous adaptation, and is no longer selected by offline configs.

`config/video_splits/*.json` contains the exact IDs. Paths resolve relative to
`datasets_root`, keeping manifests usable on another installation. Rebuild with:

```bash
python evaluation/prepare_video_splits.py /path/to/datasets
# Rebuild only the original MOSE split:
python evaluation/prepare_video_splits.py /path/to/datasets --mose-only
```

The builder rejects a wrong original-MOSE population or missing annotation video
IDs before creating the 1206/301 split. ZIP access validates frame names,
checks all available annotation/RGB alignment, and decompresses frames on demand.
YouTube-VOS ZIP CRCs, names and sizes form a recorded archive index hash;
its dense RGB frames are read on demand without archive extraction.

## Run and resume

`config/evaluation_video_dinov3.yaml` enables only the three video tasks for the
200-epoch region ViT-S/16 teacher checkpoint. General offline configs use the
same video settings and manifests. Each video task uses one GPU; all other
probe/training protocols retain their existing configurations.

```bash
sbatch slurm/evaluation_video_dinov3.sh config/evaluation_video_dinov3.yaml
# Original MOSE only:
sbatch slurm/evaluation_video_dinov3.sh config/evaluation_mose_dinov3.yaml
```

The Slurm launcher requests one GPU, four CPUs, 24GiB RAM and twelve hours.
Separate task configs allow independent scheduling. The launcher requeues near
its time limit. Atomic per-video progress is reused only when code, checkpoint,
feature selection, resolution, split manifest and archive identity match.
Historical output is preserved; launches use new output directories.

## Historical protocol comparisons

`video_protocol: dino_v1_480p` runs the pinned DINO v1 propagation and accepts
`video_feature_blocks: 1` (the upstream final-block features) or `4` (mean of
the last four normalized blocks). Both use short side 480, long side floored
to a multiple of 64, OpenCV RGB resizing, the upstream normalization including
red-channel standard deviation 0.228, temperature 0.1, top-5 neighbors across
the first frame and up to seven previous frames, and a square radius of 12
patches applied to every context frame. Mask postprocessing and soft-label
history follow the pinned evaluator. Existing `dino_480p_last4` and
`dino_square_last4` configurations retain their four-block behavior.

Only DAVIS 2017 validation is an upstream DINO benchmark. YouTube-VOS and MOSE
use explicit split manifests with the same propagation, native-resolution
J/F scoring, and first-frame objects only. YouTube-VOS results from this
extension are all-object averages, not official seen/unseen challenge scores.
Use the standard YouTube-VOS 2018 validation RGB release for the matched
three-checkpoint comparison; its sparse cadence differs from the optional
dense `all_frames` release. For this DINO comparison, an explicit
`initial_mask_root` in the split manifest points to the original supplied
initialization masks, separately from the full scoring `mask_root`. Some
full-GT first frames include objects omitted from the supplied initialization;
do not introduce those objects through scoring ground truth. Later supplied
initializations are ignored by this first-frame-only extension. Each
clip starts at its first supplied initialization; preceding unlabeled RGB
frames and their scoring annotations are excluded. The original
MOSE 2023 split is frozen at 301
evaluation clips and 1,206 disjoint reserved clips (seed 0); its author split
is unverified. Never label these results as an exact CRISP reproduction
without the corresponding split and scoring details.

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

## DINO controls on YouTube-VOS and MOSE

An explicit `video_split_manifests` entry also permits DINO propagation on
YouTube-VOS and MOSE. These are extensions: the original DINO repository releases
only a DAVIS evaluator. They retain original DINO propagation, RGB normalization,
initial mask sampling, and prediction resizing, with the requested mean of the
last four normalized blocks. `dino_480p_last4` uses a 480-pixel short side and
floors the longer side to a multiple of 64; it does not produce square inputs.

Use the exact same split manifests as the DINOv3 comparison. Propagate every
released RGB frame, initialize only objects in the first provided annotation,
and score available annotations after initialization. YouTube-VOS's unannotated
frames still update the seven-frame context. First and last frames are excluded
for DAVIS; YouTube-VOS/MOSE include the final annotation. J/F averaging is over
frames per object, then over objects. Non-DAVIS controls compute metrics without
exporting prediction masks. These settings do not establish equivalence with
CRISP's unspecified YouTube-VOS/MOSE splits or scoring adaptations.
