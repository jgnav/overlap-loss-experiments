"""CPU-only patch-ranking forward/backward benchmark with four simulated banks."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import losses.patch_rank_distribution_loss as ranking
from benchmarks.benchmark_region_ordering import simulated_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--features', type=int, default=384)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.features, args.iterations, args.threads) < 1:
        parser.error('All sizes, iterations and thread counts must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(123)
    student = tuple(torch.randn(args.batch_size, 196, args.features).requires_grad_() for _ in range(2))
    teacher = tuple(torch.randn_like(value) for value in student)
    boxes = torch.tensor([[[0., 0., 1., 1., 0.], [.125, .125, .875, .875, 1.]]] * args.batch_size)
    criterion = ranking.PatchRankDistributionLoss()
    times = []
    with mock.patch.object(ranking, 'gather_reference_bank', side_effect=simulated_bank):
        for iteration in range(args.iterations + 1):
            for value in student:
                value.grad = None
            criterion.sampling_step.zero_()
            start = time.perf_counter()
            result = criterion(student, teacher, boxes)
            result['loss'].backward()
            if iteration:
                times.append(time.perf_counter() - start)
    assert all(torch.isfinite(value.grad).all() for value in student)
    print(json.dumps({'device': 'cpu', 'batch_size': args.batch_size, 'features': args.features,
                      'simulated_ranks': 4, 'query_count': result['query_count'].item(),
                      'references_per_query': result['references_per_query'].item(),
                      'median_forward_backward_ms': round(statistics.median(times) * 1000, 3),
                      'finite_gradients': True}, indent=2))


if __name__ == '__main__':
    main()
