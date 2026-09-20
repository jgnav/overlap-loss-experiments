# Online evaluation

Training YAML files control online evaluation with:

```yaml
online_probes_enabled: true  # false disables all three tasks
online_probe_frequency: 5   # completed epochs 5, 10, 15, ...
```

Each scheduled teacher snapshot runs exactly the same offline task modules:

- `evaluation.utils.pascal_voc_knn`: original VOC train/val, CAPI parameter search, selection, refit and pixel scoring.
- `evaluation.utils.pascal_voc_linear`: offline VOC linear segmentation, including its complete parameter search.
- `evaluation.utils.imagenet_knn`: seeded stratified 10% training bank, full validation, final CLS features and temperature-weighted k-NN.

Online and offline evaluation share the worker command builder and task entrypoints.
See [the offline protocols](../evaluation/README.md) for full recipes.
Matching checkpoints and seeds use the same protocol; floating-point results can vary
with hardware and extraction batch size. The former small-subset and configurable-k
options have been removed.

Resource settings:

```yaml
online_probe_datasets_root: /path/to/datasets
online_probe_batch_size: 128       # segmentation extraction only
online_probe_num_workers: 0
online_probe_max_concurrent_jobs: 1
online_probe_wait_at_exit: true
online_probe_gpu: null             # first visible allocated GPU
```

ImageNet retains its offline batch size and neighbor settings. An explicit GPU must
name one device in `CUDA_VISIBLE_DEVICES`. Workers share an allocated GPU with
training and start an independent distributed group; no Slurm jobs are submitted.
Full offline searches take substantially more resources than the former small probes.

The interval counts completed continuation epochs. Each due epoch gets an immutable
full checkpoint copy named `teacher_epochNNNN.pth`; evaluation selects its teacher.
Busy workers queue snapshots instead of skipping them. Normal exit waits for all
queued evaluations by default. With `online_probe_wait_at_exit: false`, pending
snapshots remain for manual evaluation. Queued work is not automatically recovered
after restart. Slurm termination can interrupt running evaluations.

Artifacts under the training output directory:

- `online_probes/checkpoints/teacher_epochNNNN.pth`: immutable snapshot.
- `online_probes/logs/epochNNNN.log`: combined output of all three tasks.
- `online_probes/epochNNNN/<task>.json`: full offline results, including protocol metadata and parameter searches.
- `online_probes/epochNNNN.json`: suite status and task results.
- `online_probes/metrics.jsonl`: flattened metrics collected by training.

Metrics also go to TensorBoard and the existing W&B run at the checkpoint epoch.
Examples: `online_pascal_voc_knn_miou`, `online_pascal_voc_linear_miou`,
`online_imagenet_knn_top1` (prefixed with `train/` in W&B).
Per-task success flags and `online_probe_success` report failures.
A failed task does not prevent the other two from being attempted.

Retry within a GPU allocation with:

```bash
python -m evaluation.online_probes \
  --checkpoint /run/online_probes/checkpoints/teacher_epoch0010.pth \
  --datasets-root /path/to/datasets \
  --output /run/online_probes/epoch0010.json \
  --epoch 10 --frequency 5 --seed 0
```

Compatible completed task results are reused; stale or failed tasks rerun.
