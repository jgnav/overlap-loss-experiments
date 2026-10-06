# ViT-B and ViT-L accumulated continuation

Submit both fresh runs with `sbatch slurm/slurm_long_training_base_large.sh`.
Task 0 is B and task 1 is L. Both request four A100 or RTX PRO 6000 GPUs,
24 CPUs, 64 GiB RAM, and three days. Before timeout, Slurm requeues the task;
the trainer restores the latest epoch checkpoint, optimizer, teacher,
centers, schedule position, and original W&B ID in the same run directory.

| Setting | ViT-B | ViT-L |
| --- | --- | --- |
| Per-GPU microbatch | 48 | 32 |
| GPUs | 4 | 4 |
| Accumulation steps | 3 | 4 |
| Nominal effective batch | 576 | 512 |
| Precision | BF16 | BF16 |
| Initial actual learning rate | 7.5e-6 | 5e-6 |
| Continuation epochs | 150 | 50 |

B's 576 is the closest integer multiple of 48 × 4 to 512. The YAML learning
rates compensate for effective-batch scaling to preserve the previous runs'
initial actual learning rates. These runs start from the original pretrained
checkpoints, with fresh optimizers and W&B IDs. Region weight, geometry,
temperatures, teacher momentum, weight decay, and probe frequency are unchanged.

Accumulation defaults to one step for existing configurations. Losses are
divided by the actual number of microbatches in each window; the final short
window is trained rather than discarded. DDP synchronizes gradients on the
last microbatch. Gradient clipping, the optimizer, teacher EMA, and schedules
advance once per optimizer window. CLS and patch targets share fixed centers
throughout that window, then update their centers from its combined teacher
logits using detached FP32 sums. Epoch checkpoints contain no pending gradients
or center sums. The final short window has a smaller effective batch.

Accumulation currently excludes deep region centers, Sinkhorn, KoLeo, and
ordering objectives, whose updates or objectives depend on the microbatch.
BF16 is used without an FP16 GradScaler. Resuming with a different accumulation
factor is rejected; old checkpoints are interpreted as one accumulation step.

CPU verification compares the actual trainer's accumulated AdamW updates,
teacher EMA, and centers against full batches, including the short final window.
