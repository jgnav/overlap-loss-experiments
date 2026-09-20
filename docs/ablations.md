# Region ablations

The twelve YAML files in `config/ablations` copy `config/train.yaml` with
`additional_epochs: 50`, `online_probes_enabled: true`, and
`output_dir: output/ablation`. Each changes only its named experimental factor:

| Factor | Values |
| --- | --- |
| `region_min_area` | 0.10, 0.20, 0.30, 0.50 |
| `lambda3` | 0.20, 0.50, 1.0, 2.0 |
| `region_patch_threshold` | 0.2, 0.5, 0.8, weighted |

All other settings retain their base values, including the source checkpoint,
seed, learning rate, shared head, centering, precision, and probe interval of
five completed epochs. These are fixed config copies, not dynamically inherited
configs. The three base-value entries intentionally produce the same baseline
settings, giving twelve jobs with ten distinct experimental configurations.

With `region_patch_threshold: weighted`, a patch's weight is the fraction of
its area covered by the intersection. Pooling uses `sum(w * representation) /
sum(w)` independently in each view, for both teacher and student. Half-covered
patches have weight 0.5; fully covered patches have weight 1. Numeric values
retain threshold selection and equal weights. Empty or area-filtered pairs are
skipped. In the base centering mode this pools centered teacher probabilities
and student softmax probabilities with their existing temperatures.

Submit the full suite from the repository root:

```bash
sbatch slurm/ablation.sh
```

This creates one Slurm array with twelve independently scheduled tasks. Each
task requests one node, **four GPUs**, **24 CPUs**, **128 GB RAM**, and **50
hours**, using the partition list in the batch script. Slurm determines their
actual start times based on availability and quotas. The allocation includes
training and online probes; normal exit waits for queued probes as configured
in the base YAML.

Outputs are isolated at `output/ablation/<array_job_id>_<task_id>/`, including a
copy of the submitted config named `ablation.yaml`. Standard output/error go to
`logs/abaltion/<array_job_id>_<task_id>.out` and `.err` (the requested spelling).
Each array task starts its own training/W&B run. `squeue` may compress pending
tasks into one line; `squeue -r -j <array_job_id>` displays all twelve tasks.
