"""Prepare/run the supplied CRISP correspondence code with normalized features.

This isolates feature selection while keeping the previous native evaluation's
checkpoint, input transforms, exact sampled pairs, matching and metrics.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT / "output/analysis/crisp_native_region200_20261007_163653"
DATASETS = ("spair", "navi", "scannet")
VARIANTS = {"final_norm": 1, "last4_norm_mean": 4}
COLUMNS = {"spair": ("d0", "d1", "d2", "All"),
           "navi": ("0-30", "30-60", "60-90", "90-120"),
           "scannet": ("0-15", "15-30", "30-60", "60-180")}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare(root, baseline):
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(baseline / "source", root / "source", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    pairs = json.loads((baseline / "preflight_manifest.json").read_text())
    checkpoint = Path(pairs["checkpoint"])
    assert digest(checkpoint) == pairs["checkpoint_sha256"]
    write_json(root / "pairs.json", pairs)
    adapter = root / "source/evals/models/normalized_region_vits.py"
    shutil.copyfile(PROJECT / "evaluation/utils/crisp_normalized_features.py", adapter)
    shutil.copyfile(__file__, root / "runner.py")
    shutil.copyfile(PROJECT / "slurm/evaluation_crisp_native_features.sh", root / "launch.sh")
    for variant, blocks in VARIANTS.items():
        (root / "source/configs/backbone" / (variant + ".yaml")).write_text(
            "_target_: evals.models.normalized_region_vits.NormalizedRegionViTS\n"
            f"checkpoint_path: {checkpoint}\nfeature_blocks: {blocks}\n"
            "output: dense\nlayer: -1\n")
        for dataset in DATASETS:
            (root / variant / dataset).mkdir(parents=True)
    hashes = {str(p.relative_to(root)): digest(p) for p in (root / "source").rglob("*")
              if p.is_file() and p.suffix in (".py", ".yaml")}
    for name in ("pairs.json", "runner.py", "launch.sh"):
        hashes[name] = digest(root / name)
    write_json(root / "manifest.json", {
        "baseline_root": str(baseline), "checkpoint": str(checkpoint),
        "checkpoint_sha256": pairs["checkpoint_sha256"], "checkpoint_epoch": 200,
        "data_root": pairs["data_root"], "source_hashes": hashes,
        "feature_variants": VARIANTS,
        "feature_protocol": {
            "final_norm": "block 12 patch tokens after learned final LayerNorm",
            "last4_norm_mean": "mean of blocks 9,10,11,12, each after learned final LayerNorm",
            "projection_head": False, "softmax": False, "standard_scaler": False,
            "l2_normalization": "unchanged native matching code, after feature extraction",
            "pairs": "exact previous CRISP-native region200 pairs and ordering",
            "matching_metrics_transforms": "unchanged CRISP entry points/configs"},
        "resources": {"gpus_per_task": 1, "cpus_per_task": 4, "ram_gib": 16, "hours": 12}})
    collect(root)
    print(root, flush=True)


def pin_pairs(root):
    """Keep pair sampling identical across variants and the raw baseline."""
    from evals.datasets.spair import SPairDataset
    from evals.datasets.navi import NAVI
    from evals.datasets.scannet_pairs import ScanNetPairsDataset

    pairs = json.loads((root / "pairs.json").read_text())
    original_pairs = SPairDataset.get_pair_annotations
    original_images = SPairDataset.get_image_annotations
    original_init = SPairDataset.__init__
    pair_cache, image_cache = {}, {}

    def get_pairs(self):
        key = (self.root, self.split)
        if key not in pair_cache:
            pair_cache[key] = original_pairs(self)
        return list(pair_cache[key])

    def get_images(self):
        if self.root not in image_cache:
            image_cache[self.root] = original_images(self)
        return image_cache[self.root]

    def spair_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        class_name = kwargs["class_name"]
        difficulty = kwargs.get("vp_diff")
        expected = pairs["spair"][class_name + "/" + str(difficulty)]
        by_name = {pair["filename"]: pair for pair in get_pairs(self)}
        self.instances = [by_name[name] for name in expected]
        assert all(pair["category"] == class_name for pair in self.instances)
        if difficulty is not None:
            assert all(pair["viewpoint_variation"] == difficulty for pair in self.instances)

    SPairDataset.get_pair_annotations = get_pairs
    SPairDataset.get_image_annotations = get_images
    SPairDataset.__init__ = spair_init
    navi_init = NAVI.__init__

    def pinned_navi_init(self, *args, **kwargs):
        navi_init(self, *args, **kwargs)
        expected = pairs["navi"]
        assert set(map(tuple, self.instances)) == {tuple(row[:3]) for row in expected}
        self.instances = [tuple(row[:3]) for row in expected]
        for obj, collection, src, tgt in expected:
            self.pair_indices[obj][collection][src] = tgt

    NAVI.__init__ = pinned_navi_init
    scannet_init = ScanNetPairsDataset.__init__

    def pinned_scannet_init(self):
        scannet_init(self)
        actual = [[scene, int(src), int(tgt)] for scene, src, tgt, _ in self.instances]
        assert actual == pairs["scannet"]

    ScanNetPairsDataset.__init__ = pinned_scannet_init


def run(root, variant, dataset):
    import torch
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        assert digest(root / name) == expected, name
    assert digest(manifest["checkpoint"]) == manifest["checkpoint_sha256"]
    source = root / "source"
    sys.path.insert(0, str(source))
    os.chdir(source)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    pin_pairs(root)
    from evals.models.normalized_region_vits import NormalizedRegionViTS
    blocks = VARIANTS[variant]
    model = NormalizedRegionViTS(manifest["checkpoint"], feature_blocks=blocks).cuda().eval()
    inputs = torch.randn(1, 3, 224, 224, device="cuda")
    actual = model(inputs)
    x = model.vit.prepare_tokens(inputs)
    selected = []
    for i, block in enumerate(model.vit.blocks):
        x = block(x)
        if i >= 12 - blocks:
            selected.append(model.vit.norm(x)[:, 1:].float())
    expected = torch.stack(selected).mean(0).transpose(1, 2).reshape(1, 384, 14, 14)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all() and actual.shape == (1, 384, 14, 14)
    # Exercise the actual high-resolution patch extraction before native eval.
    sample = model(torch.zeros(2, 3, 800, 800, device="cuda"))
    assert sample.shape == (2, 384, 50, 50) and torch.isfinite(sample).all()
    report = {"variant": variant, "dataset": dataset, "feature_blocks": blocks,
              "checkpoint": manifest["checkpoint"], "shape": list(sample.shape),
              "exact_manual_layernorm_average_check": True,
              "gpu": torch.cuda.get_device_name(),
              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3}
    write_json(root / variant / dataset / "preflight.json", report)
    print("FEATURE PREFLIGHT PASSED", json.dumps(report), flush=True)
    del model, inputs, actual, expected, x, selected, sample
    torch.cuda.empty_cache()
    entry = source / ("evaluate_" + dataset + "_correspondence.py")
    sys.argv = [str(entry), "backbone=" + variant, "multilayer=false",
                "hydra.run.dir=" + str(root / variant / dataset), "hydra.job.chdir=true"]
    runpy.run_path(str(entry), run_name="__main__")


def collect(root):
    results = {}
    lines = ["# Region ViT-S epoch200: normalized correspondence features", "",
             "| Features | Dataset | Bin 1 | Bin 2 | Bin 3 | Bin 4 |",
             "|---|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        results[variant] = {}
        for dataset, columns in COLUMNS.items():
            log = root / variant / dataset / (dataset + "_correspondence.log")
            entries = [line for line in log.read_text().splitlines()
                       if "ibot_region200_vits16_" in line] if log.exists() else []
            values = [float(x.strip()) for x in entries[-1].split(",")[-4:]] if entries else None
            results[variant][dataset] = dict(zip(columns, values)) if values else None
            lines.append("| " + variant + " | " + dataset + " | " +
                         " | ".join(f"{value:.2f}" for value in values) + " |" if values else
                         "| " + variant + " | " + dataset + " | pending | pending | pending | pending |")
    write_json(root / "results.json", results)
    (root / "results_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "collect"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root, args.baseline.resolve())
    elif args.action == "collect":
        collect(root)
    else:
        if args.variant is None or args.dataset is None:
            parser.error("run requires --variant and --dataset")
        run(root, args.variant, args.dataset)
