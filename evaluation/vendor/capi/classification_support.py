# Copyright (c) Meta Platforms, Inc. and affiliates.
# Apache-2.0; see LICENSE. Pinned CAPI data and logging helpers.
from __future__ import annotations
import datetime
import itertools
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import VisionDataset
logger = logging.getLogger(__name__)
class InfiniteSampler(Sampler):
    def __init__(
        self,
        sample_count: int,
        seed: int = 0,
        advance: int = 0,
    ):
        self.sample_count = sample_count
        self.seed = seed
        self.epoch_count = advance // sample_count
        self.advance = advance - self.epoch_count * sample_count
        self.dtype = torch.int32 if self.sample_count <= 2**31 else torch.int64

    def __iter__(self) -> Iterator[int]:
        yield from itertools.islice(self._shuffled_iterator(), self.advance, None)

    def _shuffled_iterator(self) -> Iterator[int]:
        start = torch.distributed.get_rank()
        step = torch.distributed.get_world_size()
        # Always shuffle everything first
        generator = torch.Generator().manual_seed(self.seed)
        here_indices = torch.randperm(self.sample_count, dtype=self.dtype, generator=generator)[start::step]
        while True:
            # Re-seed on each iteration to allow skipping whole permutations during advance
            generator.manual_seed(self.seed + start + (self.epoch_count << 24))
            perm = torch.randperm(len(here_indices), dtype=self.dtype, generator=generator)
            yield from here_indices[perm].numpy()
            self.epoch_count += 1


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Callable[[list], Any] | None = None,
    infinite: bool = False,
) -> DataLoader:
    logger.info("Using PyTorch data loader")
    if isinstance(dataset, torch.utils.data.IterableDataset):
        logger.info("Dataset is iterable, not using a sampler")
        sampler = None
    elif infinite:
        assert shuffle, "Infinite sampler requires shuffle"
        logger.info("Using InfiniteSampler")
        sampler = InfiniteSampler(
            sample_count=len(dataset),
            seed=seed,
            advance=sampler_advance,
        )
    else:
        logger.info("Using DistributedSampler")
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    logger.info(f"batch size: {batch_size}")
    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader


class DatasetWithEnumeratedTargets(VisionDataset):
    """If pad_dataset is set, pads based on torch's DistributedSampler implementation, which
    with drop_last=False pads the last batch to be a multiple of the world size.
    https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py#L91
    """

    def __init__(self, dataset: VisionDataset, pad_dataset: bool = False, num_replicas: int | None = None):
        self._dataset = dataset
        self._size = len(self._dataset)
        self._padded_size = self._size
        self._pad_dataset = pad_dataset
        if self._pad_dataset:
            assert num_replicas is not None, "num_replicas should be set if pad_dataset is True"
            self._padded_size = num_replicas * ((len(dataset) + num_replicas - 1) // num_replicas)

    def __getitem__(self, index: int) -> tuple[Any, tuple[int, int]]:
        image, target = self._dataset[index % self._size]
        if index >= self._size:
            assert self._pad_dataset
            return image, (-1, target)
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        return self._padded_size


class MetricLogger:
    def __init__(self, delimiter: str = "  ", output_file: str | Path | None = None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        if isinstance(output_file, str):
            output_file = Path(output_file)
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter!s}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or torch.distributed.get_rank() != 0:
            return
        dict_to_dump = {
            "iteration": iteration,
            "iter_time": iter_time,
            "data_time": data_time,
        }
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with self.output_file.open("a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}MB"]

        log_msg = self.delimiter.join(log_list)
        if i < n_iterations:
            for obj in iterable:
                data_time.update(time.time() - end)
                yield obj
                iter_time.update(time.time() - end)
                if i % print_freq == 0 or i == n_iterations - 1:
                    self.synchronize_between_processes()
                    self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                    eta_seconds = iter_time.global_avg * (n_iterations - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        logger.info(
                            log_msg.format(
                                i,
                                n_iterations,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            ),
                        )
                    else:
                        logger.info(
                            log_msg.format(
                                i,
                                n_iterations,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                            ),
                        )
                i += 1
                end = time.time()
                if i >= n_iterations:
                    break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"{header} Total time: {total_time_str} ({total_time / n_iterations:.6f} s / it)")


def to_tensor(x: Tensor | float | int) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.tensor(x)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.window_size = window_size
        self.deque: deque[Tensor | float | int] = deque(maxlen=window_size)
        self.total: Tensor | float | int = 0.0
        self.count: int = 0
        self.fmt = fmt

    def update(self, value: Tensor | float | int):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def synchronize_between_processes(self):
        """Distributed synchronization of the metric"""
        if not torch.distributed.is_initialized():
            return
        logger.debug("Synchronizing values")
        count = to_tensor(self.count).to(dtype=torch.float64, device="cuda").reshape(1)
        total = to_tensor(self.total).to(dtype=torch.float64, device="cuda").reshape(1)
        tensor_deque = torch.tensor(list(self.deque), dtype=torch.float64, device="cuda")
        t = torch.cat([count, total, tensor_deque], dim=0)
        torch.distributed.barrier()
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
        self.count = int(t[0].cpu().item())
        self.total = t[1]
        self.deque = deque(list(t[2:]), maxlen=self.window_size)

    @property
    def median(self) -> float | int:
        d = torch.tensor(list(self.deque))
        return d.median().cpu().item()

    @property
    def avg(self) -> float | int:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().cpu().item()

    @property
    def global_avg(self) -> float | int:
        return to_tensor(self.total).cpu().item() / self.count

    @property
    def max(self) -> float | int:
        return torch.tensor(self.deque).max().cpu().item()

    @property
    def value(self) -> float | int:
        v = self.deque[-1]
        return to_tensor(v).cpu().item()

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )

