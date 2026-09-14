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
not change between epochs. `online_probe_gpu` can name a free GPU in a separate
allocation. `online_probe_max_concurrent_jobs` prevents an unbounded backlog;
if all slots are occupied, that epoch is recorded as skipped in the training
output, while its immutable snapshot is retained for a later manual retry.
Probe stdout/stderr is retained under
`<run>/online_probes/logs/`, while completed results are written atomically as
`epochNNNN.json`.

Completed results are picked up by the existing training statistics path before
the next training record is written. The metrics therefore appear in TensorBoard, JSON and
W&B with names such as `online_imagenet_cls_knn_top1` and
`online_voc_dense_knn_miou_percent`, together with
`online_probe_completed_epoch`. The per-epoch JSON remains the authoritative
record when a worker finishes after training has already ended.

The standalone worker is useful for retrying a copied checkpoint without
touching the mutable training checkpoint:

```bash
python -m evaluation.online_probes \
  --checkpoint output/run/online_probes/checkpoints/teacher_epoch0010.pth \
  --datasets-root dataset \
  --output output/run/online_probes/epoch0010.json
```

Probe failures do not modify model state or loss computation. They are visible
in the corresponding worker log and can be rerun from the immutable snapshot.
