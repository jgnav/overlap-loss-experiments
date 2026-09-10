"""Local I/O adapters for the pinned CAPI segmentation evaluator.

Distributed extraction uses the same final patch features and row-major pixel
labels as CAPI's extract_features. No classifier or scoring logic lives here.
"""

from sklearn.preprocessing import StandardScaler
from torchvision.transforms import Normalize

from evaluation.utils.common import print_progress


CAPI_REVISION = "98b4fa17ee8eec8810c17022df9a27a44845368b"
IMAGENET_NORM = Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
)
standardizations = {"StandardScaler": StandardScaler}


def make_dataset(dataset_str_or_path, transform=None, target_transform=None):
    # Our already-resolved datasets preserve local file order and VOC train/val.
    dataset = dataset_str_or_path
    dataset.transform = transform
    dataset.target_transform = target_transform
    return dataset


class MetricLogger:
    def log_every(self, iterable, print_freq=10, header=""):
        for index, value in enumerate(iterable, start=1):
            yield value
            print_progress(header, index, len(iterable))


def extract_features(model, dataset, batch_size, num_workers, *, gather_on_cpu=False):
    from evaluation.utils.dense import _extract_features
    from evaluation.utils.distributed_features import gather_image_shards
    import torch.distributed as dist
    from torch.utils.data import Subset

    if not gather_on_cpu:
        raise ValueError("CAPI extraction requires CPU feature storage")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    shard = Subset(dataset, range(rank, len(dataset), world_size))
    features = labels = None
    if len(shard):
        features, labels = _extract_features(
            model, shard, batch_size, num_workers, f"CAPI features rank {rank}"
        )
        features = features.reshape(len(shard), 16, 16, -1)
        labels = labels.reshape(len(shard), 16, 16, -1)
    if world_size == 1:
        return features, labels
    return gather_image_shards(features, labels, len(dataset))
