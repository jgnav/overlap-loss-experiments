"""Compatibility for the three online probes, independent of other benchmarks."""

import ast
import hashlib
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_EVALUATIONS = ("pascal_voc_knn", "pascal_voc_linear", "imagenet_knn")


def _selected_source(path, names):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
    found = {node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    found.update(target.id for node in nodes if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name))
    if found != set(names):
        raise ValueError(f"Missing probe dependencies in {path}: {set(names) - found}")
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()


def probe_source_hash(evaluation_name, repository_root=REPO_ROOT):
    """Hash probe dependencies; omit unrelated tasks, CLI metadata and reporting."""
    if evaluation_name not in ONLINE_EVALUATIONS:
        raise ValueError(f"Not an online probe: {evaluation_name}")
    root = Path(repository_root)
    files = ["model/vision_transformer.py", "evaluation/utils/online_probe_compatibility.py"]
    selected = {
        "model/__init__.py": ("ARCHITECTURES", "create_model"),
        "utils/training.py": ("trunc_normal_", "_no_grad_trunc_normal_"),
        "evaluation/utils/common.py": (
            "ARCHITECTURES", "_torch_load", "_checkpoint_state", "_canonical_backbone_state",
            "_checkpoint_argument", "_infer_architecture", "load_backbone", "initialize_distributed",
        ),
    }
    if evaluation_name.startswith("pascal_voc"):
        files += ["evaluation/utils/dense.py", "evaluation/utils/capi_adapter.py",
                  "evaluation/utils/distributed_features.py", "evaluation/vendor/capi/eval_segmentation.py"]
        selected["evaluation/utils/datasets.py"] = (
            "_read_ids", "_resolve_voc_root", "SegmentationDataset", "make_pascal_voc", "segmentation_manifest",
        )
    else:
        files += ["evaluation/utils/imagenet.py"]
    files += [f"evaluation/utils/{evaluation_name}.py"]
    digest = hashlib.sha256()
    for relative in sorted(files + list(selected)):
        path = root / relative
        content = _selected_source(path, selected[relative]) if relative in selected else path.read_bytes()
        digest.update(relative.encode())
        digest.update(content)
    return digest.hexdigest()


def _dataset_inputs(args, evaluation_name):
    if evaluation_name.startswith("pascal_voc"):
        from evaluation.utils.datasets import _resolve_voc_root
        root = _resolve_voc_root(Path(args.datasets_root))
        return {split: hashlib.sha256((root / "ImageSets/Segmentation" / f"{split}.txt").read_bytes()).hexdigest()
                for split in ("train", "val")}
    path = REPO_ROOT / "evaluation/resources/simclrv2_imagenet_subsets/10percent.txt"
    return {"subset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def online_probe_identity(args, evaluation_name):
    return {
        "version": 1, "evaluation": evaluation_name,
        "source_sha256": probe_source_hash(evaluation_name),
        "seed": args.seed, "architecture_argument": args.arch,
        "checkpoint_key": args.checkpoint_key,
        "datasets_root": str(Path(args.datasets_root).expanduser().resolve()),
        "dataset_inputs": _dataset_inputs(args, evaluation_name),
    }


def _protocol_matches(result, args, evaluation_name):
    model = result.get("model", {})
    protocol = result.get("protocol", {})
    if evaluation_name.startswith("pascal_voc"):
        from evaluation.utils.capi_adapter import CAPI_REVISION
        from evaluation.utils.datasets import make_pascal_voc, segmentation_manifest
        patch_size = model.get("patch_size")
        if patch_size not in (14, 16):
            return False
        expected = {
            "capi_revision": CAPI_REVISION, "dataset_train_split": "train", "dataset_test_split": "val",
            "input_resolution": 16 * patch_size, "patch_tokens": 256, "knn_dtype": "float32",
            "backbone_frozen": True, "feature": f"final normalized {args.checkpoint_key} patch tokens",
            "standardization": "StandardScaler fitted on train only", "validation_split": "seeded 10% of training set",
            "num_classes": 21, "ignore_labels": [255], "gpu_count": 1,
        }
        manifests = {key: segmentation_manifest(make_pascal_voc(Path(args.datasets_root), split))
                     for key, split in (("train", "train"), ("test", "val"))}
        saved = result.get("dataset_manifests", {})
        for key, current in manifests.items():
            for field in ("ordered_pairs_sha256", "samples", "construction", "mask_policy", "training_order"):
                if saved.get(key, {}).get(field) != current.get(field):
                    return False
        classifier = "knn" if evaluation_name.endswith("knn") else "linear_logistic_regression"
        if result.get("classifier") != classifier:
            return False
    else:
        expected = {
            "input_resolution": 224, "training_fraction": 0.1,
            "training_subset_file_sha256": _dataset_inputs(args, evaluation_name)["subset_sha256"],
            "feature": f"final normalized {args.checkpoint_key} CLS token", "feature_l2_normalization": True,
            "temperature": 0.07, "neighbors": [10, 20, 100, 200], "primary_neighbors": 20,
            "gpu_count": 1, "batch_size_per_gpu": 256,
        }
    return all(protocol.get(key) == value for key, value in expected.items())


def load_online_probe_result(path, args, evaluation_name, *, audited_legacy_hashes=()):
    """Require relevant identity; legacy migration requires an explicit source audit."""
    from evaluation.utils.common import checkpoint_fingerprint
    try:
        result = json.loads(Path(path).read_text())
        model = result.get("model", {})
        if (result.get("status") != "completed" or result.get("evaluation") != evaluation_name
                or model.get("checkpoint_key") != args.checkpoint_key
                or model.get("checkpoint_fingerprint") != checkpoint_fingerprint(args.checkpoint)):
            return None
        metric = "top1" if evaluation_name == "imagenet_knn" else "miou_percent"
        value = result.get("metrics", {}).get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        if args.arch != "auto" and model.get("architecture") != args.arch:
            return None
        current = online_probe_identity(args, evaluation_name)
        saved = result.get("online_probe_identity")
        if saved is not None:
            if saved != current:
                return None
        else:
            legacy = result.get("evaluation_identity", {})
            if (legacy.get("source_sha256") not in audited_legacy_hashes
                    or any(legacy.get(key) != current[key]
                           for key in ("seed", "architecture_argument", "datasets_root"))):
                return None
        if not _protocol_matches(result, args, evaluation_name):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None
