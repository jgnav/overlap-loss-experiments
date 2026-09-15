# Online representation probes

The optional probes in `evaluation/online_probes.py` monitor the EMA teacher
during continuation training. They use one fixed nearest-neighbour protocol,
so a change in the reported score reflects the representation rather than a
new hyperparameter search.

Set `online_probes_enabled: true` in `config/train.yaml` to enable them. Every
`online_probe_frequency` completed epoch (10 by default), rank zero makes an
atomic copy of the just-written `checkpoint.pth` under
`<run>/online_probes/checkpoints/` and launches a bounded asynchronous worker:

* ImageNet CLS k-NN uses a deterministic class-stratified bank of 10,000 train
  images and 5,000 validation images, a bicubic resize followed by a 224 x 224
  center crop, the teacher's final normalized CLS token, cosine similarity and
  fixed `k=20`. It reports top-1 and top-5.
* VOC dense k-NN uses fixed seeded subsets of 400 train and 200 validation
  images, a 256 x 256 resize, final normalized 16 x 16 patch tokens from the
  teacher, cosine similarity and fixed `k=20`. Nearest patch votes are tiled
  back to pixels and scored with mIoU and pixel accuracy. Ignore label 255 is
  excluded from both metrics.

The subsets, seed, layer, input sizes and k are passed to every worker and do
not change between epochs. Workers run locally with `subprocess.Popen`; this
does **not** request another Slurm allocation. With `online_probe_gpu: null`,
the worker shares the first visible training GPU. An explicit GPU ID only
selects an already allocated GPU on the same host. A separate allocation needs
its own Slurm submission running the standalone worker below.
`online_probe_max_concurrent_jobs` prevents an unbounded backlog;
if all slots are occupied, that epoch is recorded as skipped in the training
output, while its immutable snapshot is retained for a later manual retry.
Probe stdout/stderr is retained under
`<run>/online_probes/logs/`, while completed results are written atomically as
`epochNNNN.json`.

Completed results are picked up at training epoch boundaries and once more at
shutdown. Every result gets its own record in `online_probes/metrics.jsonl`,
TensorBoard and W&B, so two workers finishing together do not overwrite each
other. W&B names include `train/online_imagenet_cls_knn_top1` and
`train/online_voc_dense_knn_miou_percent`. Their horizontal axis is
`train/online_probe_epoch`, the evaluated checkpoint's completed continuation
epoch, rather than the epoch when the result was collected.
`train/online_probe_success` is 1 on success and 0 on failure; errors are also
printed in the training log. A worker killed without a result file is reported
as failed when the parent observes its exit. The per-epoch JSON remains the authoritative
record when a worker finishes after training has already ended.

The standalone worker is useful for retrying a copied checkpoint without
touching the mutable training checkpoint:

```bash
python -m evaluation.online_probes \
  --checkpoint output/run/online_probes/checkpoints/teacher_epoch0010.pth \
  --datasets-root /mnt/fast/nobackup/scratch4weeks/jg02228/datasets \
  --output output/run/online_probes/epoch0010.json
```

Probe failures do not modify model state or loss computation. They are visible
in the corresponding worker log and can be rerun from the immutable snapshot.
Standalone workers write JSON; they do not independently upload to W&B. Results
that finish after the training process has exited need a separate import.

## Dataset paths

The training `data_path` must be the ImageNet **training split**, while
`online_probe_datasets_root` is the parent containing both datasets:

```yaml
data_path: /mnt/fast/nobackup/scratch4weeks/jg02228/datasets/imagenet/train
online_probe_datasets_root: /mnt/fast/nobackup/scratch4weeks/jg02228/datasets
online_probe_frequency: 10
```

The expected layout is `imagenet/{train,val}/<synset>/*.JPEG` and
`pascal_voc/VOCdevkit/VOC2012/` with the original segmentation split lists,
JPEG images and PNG masks. Startup checks reject missing ImageNet splits,
mismatched class directories, and missing VOC files before training begins.
Workers repeat these checks before loading a model. Their input and output
paths are absolute so changing the worker's working directory is harmless.

## Audit of job 55340 (15 September 2026)

The feature protocol, subset sizes, crop sizes, teacher selection and immutable
snapshots matched the requested probes. Corrections made after this job started:

* The training configs used frequency 5 and the nonexistent relative path
  `dataset`; they now use frequency 10 and the shared dataset root above.
* The training input was the parent of all datasets. Its loader reported
  1,485,725 images, including other datasets, rather than restricting itself to
  ImageNet training images. The configs now select `imagenet/train`.
* Failed probes were silently dropped from training metrics. Simultaneously
  completed probes could overwrite each other in a single dictionary. Logging
  now preserves individual records and reports failures.
* Class votes used nested Python loops and similarity search always ran on
  CPU. Votes are now batched and similarity search uses the probe device,
  still with bounded query chunks and the same fixed voting rule.

The dataset is not ready for an end-to-end probe: `imagenet/val` was absent at
audit time. Preparation job 54884 ended with a bus error after logging
1,280,000 training images and zero validation images. VOC split lists contain
1,464 training and 1,449 validation entries. Changing paths does not complete
ImageNet extraction; the new preflight deliberately reports that condition.

These edits do not update configuration or imported training code inside job
55340. The running training job was not stopped or restarted. GPU contention,
full-data runtime, and end-to-end scores still need validation once ImageNet
preparation is complete.

Validation: all 13 online-probe regression tests passed. The epoch-10 teacher
snapshot from job 55340 loaded as ViT-S/16 and produced finite final tokens of
shape `[1, 197, 384]` at 224 pixels and `[1, 257, 384]` at 256 pixels on CPU.
The broader training-config suite passed 12 of 13 tests; its existing fixed
batch-size assertion expects 256 while the current config uses 384. That
training batch-size setting was left unchanged.
