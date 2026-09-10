"""Validated run configuration for evaluation.py; probe recipes remain fixed."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from evaluation.utils.common import ARCHITECTURES
from evaluation.utils.orchestrator import EVALUATIONS
from utils.wandb_logging import WANDB_DEFAULTS, configure_wandb


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently replacing a user's selection."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("Evaluation YAML keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate evaluation configuration key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        values = yaml.load(handle, Loader=_UniqueKeyLoader)
    if not isinstance(values, dict):
        raise ValueError("Evaluation configuration must be a YAML mapping")
    defaults = {
        **WANDB_DEFAULTS,
        "checkpoint_key": "teacher", "arch": "auto", "num_workers": 8,
        "seed": 0, "segmentation_batch_size": 128,
        "output_dir": None, "result_json": None, "classification_manifests": None,
    }
    required = {"checkpoint", "datasets_root", "evaluations"}
    unknown = set(values) - required - set(defaults)
    if unknown:
        raise ValueError(f"Unknown evaluation configuration keys: {', '.join(sorted(unknown))}")
    missing = required - set(values)
    if missing:
        raise ValueError(f"Missing evaluation configuration keys: {', '.join(sorted(missing))}")
    values = {**defaults, **values}
    configure_wandb(values)
    if values["checkpoint_key"] not in ("teacher", "student"):
        raise ValueError("checkpoint_key must be teacher or student")
    if values["arch"] not in ("auto", *ARCHITECTURES):
        raise ValueError(f"arch must be auto or one of {ARCHITECTURES}")
    for name, minimum in (("num_workers", 0), ("seed", 0), ("segmentation_batch_size", 1)):
        if type(values[name]) is not int or values[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if values["seed"] >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    switches = values["evaluations"]
    if not isinstance(switches, dict):
        raise ValueError("evaluations must map evaluation names to true or false")
    names = [name for name, _, _ in EVALUATIONS]
    unknown = set(switches) - set(names)
    if unknown:
        raise ValueError(f"Unknown evaluations: {', '.join(sorted(unknown))}")
    if any(type(enabled) is not bool for enabled in switches.values()):
        raise ValueError("Evaluation switches must be YAML true or false, not strings or numbers")
    values["evaluations"] = [name for name in names if switches.get(name, False)]
    if not values["evaluations"]:
        raise ValueError("Enable at least one evaluation in the YAML")
    for name in ("checkpoint", "datasets_root", "output_dir", "result_json", "classification_manifests"):
        value = values[name]
        if value is None and name not in ("checkpoint", "datasets_root"):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        value = Path(value).expanduser()
        values[name] = (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
    values["config_path"] = path
    return SimpleNamespace(**values)


def config_snapshot(args):
    """Save effective paths and every switch, including omitted/disabled tasks."""
    keys = ("checkpoint", "checkpoint_key", "arch", "datasets_root", "classification_manifests",
            "output_dir", "result_json", "seed", "num_workers", "segmentation_batch_size", *WANDB_DEFAULTS)
    result = {key: str(value) if isinstance(value := getattr(args, key), Path) else value for key in keys}
    result["evaluations"] = {name: name in args.evaluations for name, _, _ in EVALUATIONS}
    return result
