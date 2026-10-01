"""CPU-only loss benchmark; optionally compare a saved pre-optimization source.

python benchmarks/benchmark_region_ordering.py --reference-source /tmp/region-ordering-before
The simulated cross-image bank has four ranks; no GPU or backbone is used.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import losses.region_ordering_loss as optimized


def reference_module(directory):
    def read(name, filename):
        spec = importlib.util.spec_from_file_location(name, directory / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    sorter = read("losses._benchmark_reference_sorter", "region_sorting.py")
    ordering = read("losses._benchmark_reference_ordering", "region_ordering_loss.py")
    ordering.permutation_cross_entropy = sorter.permutation_cross_entropy
    return ordering


def make_inputs(batch, dimensions):
    generator = torch.Generator().manual_seed(123)
    sizes = (196, 196) + (36,) * 10
    student = tuple((torch.randn(batch, size, dimensions, generator=generator) * .05).requires_grad_()
                    for size in sizes)
    targets = tuple((torch.randn(batch, 196, dimensions, generator=generator) * .05 / .07).softmax(-1)
                    for _ in range(2))
    crops = [[0., 0., 1., 1., 0.], [.125, .125, .875, .875, 1.]]
    for index in range(10):
        left, top = (index % 4) * .125, (index // 4) * .125
        crops.append([left, top, left + .5, top + .5, float(index % 2)])
    return student, targets, torch.tensor([crops] * batch)


def simulated_bank(bank, owners, gather_vectors=True):
    banks = [bank.roll(rank, 0) for rank in range(4)]
    ids = [torch.stack((torch.full_like(owners[:, 0], rank), owners[:, 1]), -1) for rank in range(4)]
    return torch.cat(banks), torch.cat(ids)


def measure(module, modality, inputs, iterations, profile):
    criterion = module.RegionOrderingLoss(modality, seed=0, min_area=.1)
    student, targets, boxes = inputs
    def step():
        for view in student:
            view.grad = None
        criterion.sampling_step.zero_()
        result = criterion(student, targets, boxes)
        result["loss"].backward()
        return result
    with mock.patch.object(module, "gather_reference_bank", side_effect=simulated_bank):
        step()
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = step()
            times.append(time.perf_counter() - start)
        operations = {}
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                step()
            watched = {"aten::_softmax", "aten::atan", "aten::index", "aten::index_copy", "aten::mm", "aten::bmm"}
            operations = {event.key: event.count for event in trace.key_averages() if event.key in watched}
        gradients = tuple(None if view.grad is None else view.grad.clone() for view in student)
    return {
        "median_forward_backward_ms": round(statistics.median(times) * 1000, 3),
        "query_count": result["query_count"].item(),
        "references_per_query": result["references_per_query"].item(),
        "cpu_operator_counts": operations,
    }, result["loss"].detach(), gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prototypes", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--modality", choices=("all", "cross_image", "within_image"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    inputs = make_inputs(args.batch_size, args.prototypes)
    modules = {"optimized": optimized}
    if args.reference_source:
        modules = {"before": reference_module(args.reference_source), **modules}
    output = {"device": "cpu", "batch_size": args.batch_size, "prototypes": args.prototypes,
              "threads": args.threads, "simulated_cross_image_ranks": 4, "results": {}}
    modalities = ("cross_image", "within_image") if args.modality == "all" else (args.modality,)
    for modality in modalities:
        results, comparisons = {}, {}
        for name, module in modules.items():
            result, loss, gradients = measure(module, modality, inputs, args.iterations, args.profile)
            results[name] = result
            comparisons[name] = (loss, gradients)
        if "before" in comparisons:
            expected, actual = comparisons["before"], comparisons["optimized"]
            torch.testing.assert_close(actual[0], expected[0], atol=2e-4, rtol=2e-4)
            for gradient, reference in zip(actual[1], expected[1]):
                if reference is None:
                    assert gradient is None
                else:
                    torch.testing.assert_close(gradient, reference, atol=2e-5, rtol=5e-3)
            results["loss_and_gradients_match"] = True
            results["speedup"] = round(results["before"]["median_forward_backward_ms"] /
                                       results["optimized"]["median_forward_backward_ms"], 2)
        output["results"][modality] = results
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
