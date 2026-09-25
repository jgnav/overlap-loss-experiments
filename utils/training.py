import builtins
import datetime
import math
import os
import random
import sys
import tempfile
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import ImageFilter, ImageOps


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Original iBOT/timm truncated-normal initializer."""

    def norm_cdf(value):
        return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image):
        if random.random() > self.prob:
            return image
        return image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return ImageOps.solarize(image)
        return image


def clip_gradients(model, clip):
    """Clip each parameter gradient exactly as in the original iBOT trainer."""
    norms = []
    for _, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter_norm = parameter.grad.data.norm(2)
            norms.append(parameter_norm.item())
            clip_coefficient = clip / (parameter_norm + 1e-6)
            if clip_coefficient < 1:
                parameter.grad.data.mul_(clip_coefficient)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for name, parameter in model.named_parameters():
        if "last_layer" in name:
            parameter.grad = None


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    iterations_per_epoch,
    warmup_epochs=0,
    start_warmup_value=0,
):
    warmup_schedule = np.array([])
    warmup_iterations = warmup_epochs * iterations_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iterations
        )
    iterations = np.arange(epochs * iterations_per_epoch - warmup_iterations)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iterations / len(iterations))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * iterations_per_epoch
    return schedule


def fix_random_seeds(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        value = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(value)
        value = value.tolist()
        self.count = int(value[0])
        self.total = value[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[key].update(value)

    def __getattr__(self, attribute):
        if attribute in self.meters:
            return self.meters[attribute]
        if attribute in self.__dict__:
            return self.__dict__[attribute]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attribute}'"
        )

    def __str__(self):
        return self.delimiter.join(
            f"{name}: {meter}" for name, meter in self.meters.items()
        )

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_frequency, header=None):
        index = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iteration_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_format = ":" + str(len(str(len(iterable)))) + "d"
        log_message = self.delimiter.join(
            [
                header,
                "[{0" + space_format + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ]
        )
        megabyte = 1024.0 * 1024.0
        for item in iterable:
            data_time.update(time.time() - end)
            yield item
            iteration_time.update(time.time() - end)
            if index % print_frequency == 0 or index == len(iterable) - 1:
                eta_seconds = iteration_time.global_avg * (len(iterable) - index)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    log_message.format(
                        index,
                        len(iterable),
                        eta=eta,
                        meters=str(self),
                        time=str(iteration_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / megabyte,
                    )
                )
            index += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_string = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f"{header} Total time: {total_time_string} "
            f"({total_time / len(iterable):.6f} s / it)"
        )


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, path):
    """Atomically replace a checkpoint after the complete write succeeds."""
    if not is_main_process():
        return
    destination = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def setup_for_distributed(is_master):
    builtin_print = builtins.print

    def distributed_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = distributed_print


def init_distributed_mode(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if args.gpu_count > 1:
            raise RuntimeError(
                f"This configuration requires {args.gpu_count} processes. Launch with: "
                f"torchrun --standalone --nproc-per-node={args.gpu_count} "
                "train.py <config.yaml>"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The original iBOT trainer requires CUDA")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    if args.world_size != args.gpu_count:
        raise RuntimeError(
            f"torchrun started {args.world_size} processes, but the YAML requests "
            f"{args.gpu_count} GPUs"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The original iBOT trainer requires CUDA")
    if args.gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={args.gpu}, but only {torch.cuda.device_count()} GPUs are visible"
        )

    dist.init_process_group(
        backend=args.distributed_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.gpu)
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class MultiCropWrapper(nn.Module):
    """The multi-resolution forward wrapper from DINO/iBOT."""

    def __init__(self, backbone, head=None):
        super().__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = nn.Identity() if head is None else head

    @staticmethod
    def _pool_patch_softmax(patches, weights, project, temperature, chunk_size=16):
        """Pool logits by patch chunk; student autograd still saves activations."""
        if weights.shape != patches.shape[:2]:
            raise ValueError("Deep region weights must match patch-token shape")
        weights = weights.float()
        # Invalid regions are discarded by the loss; give them a finite
        # temporary representation so no log(0) reaches autograd.
        empty = weights.sum(dim=1) == 0
        if empty.any():
            weights = weights.clone()
            weights[empty, 0] = 1
        pooled = None
        for offset in range(0, patches.shape[1], chunk_size):
            chunk_weights = weights[:, offset:offset + chunk_size]
            no_selected_patch = chunk_weights.sum(dim=1) == 0
            if no_selected_patch.any():
                safe_weights = chunk_weights.clone()
                safe_weights[no_selected_patch, 0] = 1
            else:
                safe_weights = chunk_weights
            logits = project(patches[:, offset:offset + chunk_size])
            logs = nn.functional.log_softmax(logits.float() / temperature, dim=-1)
            chunk = torch.logsumexp(
                logs + safe_weights.log().unsqueeze(-1), dim=1
            )
            # A finite sentinel prevents undefined gradients in logaddexp when
            # an entire patch chunk is outside a sample's region.
            chunk = chunk.masked_fill(no_selected_patch[:, None], -1e9)
            pooled = chunk if pooled is None else torch.logaddexp(pooled, chunk)
        return pooled - weights.sum(dim=1, keepdim=True).log()

    def forward(self, inputs, mask=None, return_backbone_feat=False,
                return_deep_patches=False, deep_region_weights=None,
                deep_softmax_temperature=None, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
            mask = [mask] if mask is not None else None
        if deep_softmax_temperature is not None and (
            not return_deep_patches or deep_region_weights is None
        ):
            raise ValueError("Deep softmax requires intermediate patches and region weights")
        crop_boundaries = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([item.shape[-1] for item in inputs]),
                return_counts=True,
            )[1],
            0,
        )
        start_index = 0
        outputs = []
        deep_outputs = {depth: [] for depth in (3, 6, 9, 12)} if return_deep_patches else None
        for end_index in crop_boundaries:
            input_batch = torch.cat(inputs[start_index:end_index])
            if mask is not None:
                input_mask = torch.cat(mask[start_index:end_index])
                kwargs.update(mask=input_mask)
            current = (
                self.backbone(input_batch, return_deep_patches=True, **kwargs)
                if return_deep_patches else self.backbone(input_batch, **kwargs)
            )
            if return_deep_patches:
                current_output, current_deep = current
                for depth in deep_outputs:
                    deep_outputs[depth].append(current_deep[depth])
            else:
                current_output = current
            outputs.append(current_output)
            start_index = end_index
        output = torch.cat(outputs)
        register_count = getattr(self.backbone, "num_register_tokens", 0)
        head_input = torch.cat(
            (output[:, :1], output[:, 1 + register_count:]), dim=1
        )
        projected_output = self.head(head_input)
        if return_deep_patches:
            deep_outputs = {
                depth: torch.cat(chunks) for depth, chunks in deep_outputs.items()
            }
            if deep_softmax_temperature is not None:
                weights = torch.cat(deep_region_weights, dim=0)
                deep_outputs = {
                    depth: self._pool_patch_softmax(
                        projected_output[1] if depth == 12 else patches,
                        weights,
                        nn.Identity() if depth == 12 else self.head.forward_patches,
                        deep_softmax_temperature,
                    )
                    for depth, patches in deep_outputs.items()
                }
            if return_backbone_feat:
                return output, projected_output, deep_outputs
            return projected_output, deep_outputs
        if return_backbone_feat:
            return output, projected_output
        return projected_output


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or len(parameter.shape) == 1:
            not_regularized.append(parameter)
        else:
            regularized.append(parameter)
    return [
        {"params": regularized},
        {"params": not_regularized, "weight_decay": 0.0},
    ]


def has_batchnorms(model):
    batch_norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    return any(
        isinstance(module, batch_norm_types) for _, module in model.named_modules()
    )


@torch.no_grad()
def concat_all_gather(tensor):
    gathered = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor, async_op=False)
    return torch.cat(gathered, dim=0)


def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()
