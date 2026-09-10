"""Gather strided image shards into their original order with bounded transfers."""

import math

import torch
import torch.distributed as dist

MAX_TRANSFER_BYTES = 16 * 1024**2


def gather_image_shards(features, labels, image_count):
    """Replicate CPU features/labels for CAPI's distributed classifier sweep.

    Rank r owns images r, r+world_size, ... . Empty ranks pass None. Only small
    shape metadata uses object collectives; tensors travel in <=16 MiB chunks
    (or one image, if larger) instead of pickling entire feature banks on CUDA.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    metadata = [(features.shape[1:], features.dtype, labels.shape[1:], labels.dtype)
                if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    feature_shape, feature_dtype, label_shape, label_dtype = metadata[0]
    all_features = torch.empty((image_count, *feature_shape), dtype=feature_dtype)
    all_labels = torch.empty((image_count, *label_shape), dtype=label_dtype)
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    for local, output, shape, dtype in (
        (features, all_features, feature_shape, feature_dtype),
        (labels, all_labels, label_shape, label_dtype),
    ):
        image_bytes = math.prod(shape) * output.element_size()
        chunk_images = max(1, MAX_TRANSFER_BYTES // image_bytes)
        for owner in range(world_size):
            count = len(range(owner, image_count, world_size))
            for start in range(0, count, chunk_images):
                end = min(count, start + chunk_images)
                buffer = torch.empty((end - start, *shape), dtype=dtype, device=device)
                if rank == owner:
                    buffer.copy_(local[start:end])
                dist.broadcast(buffer, src=owner)
                output[owner + start * world_size:owner + end * world_size:world_size].copy_(buffer.cpu())
    return all_features, all_labels
