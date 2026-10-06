"""Submit an offline suite as independent Slurm jobs with task-specific resources."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
GROUPS = {
    "segmentation": ({"pascal_voc_knn", "pascal_voc_linear", "ade20k_knn", "ade20k_linear",
                      "cityscapes_knn", "cityscapes_linear"}, 1, 16, 128, "2-00:00:00"),
    "imagenet_linear": ({"imagenet_linear"}, 4, 24, 64, "3-00:00:00"),
    "classification": ({"imagenet_knn", "imagenet_knn_1pct", "imagenet_knn_100pct",
                        "pascal_voc_multilabel", "pascal_voc_1shot", "pascal_voc_2shot",
                        "pascal_voc_5shot", "coco_multilabel", "visual_genome_multilabel"},
                       4, 24, 64, "1-00:00:00"),
    "benchmarks": ({"spair_correspondence", "navi_correspondence", "scannet_correspondence",
                    "davis_vos", "youtube_vos_vos", "mose_vos"}, 1, 8, 32, "3-00:00:00"),
}


def prepare(config_path, output_dir=None):
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text())
    switches = config.get("evaluations", {})
    allowed = set().union(*(spec[0] for spec in GROUPS.values()))
    if not switches or set(switches) - allowed or any(type(v) is not bool for v in switches.values()):
        raise ValueError("Expected nonempty evaluation switches with known names and boolean values")
    if not any(switches.values()):
        raise ValueError("Enable at least one evaluation")
    for key in ("checkpoint", "datasets_root", "classification_manifests"):
        if config.get(key) is not None:
            value = Path(config[key]).expanduser()
            config[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    if not Path(config["checkpoint"]).is_file() or not Path(config["datasets_root"]).is_dir():
        raise FileNotFoundError("Checkpoint or datasets root does not exist")
    target = config.get("output_dir")
    if output_dir is not None:
        root = Path(output_dir).expanduser().resolve()
    elif target:
        target = Path(target).expanduser()
        root = (path.parent / target).resolve() if not target.is_absolute() else target.resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = REPO / "output/evaluation" / f"{Path(config['checkpoint']).stem}-crisp-{timestamp}"
    return config, root


def submit(config_path, output_dir=None, dry_run=False):
    config, root = prepare(config_path, output_dir)
    jobs = []
    for group, (names, gpus, cpus, memory, wall_time) in GROUPS.items():
        selected = [name for name, enabled in config["evaluations"].items() if enabled and name in names]
        if selected:
            jobs.append({"group": group, "evaluations": selected, "gpu_count": gpus,
                         "cpus": cpus, "memory_gib": memory, "time_limit": wall_time,
                         "run_dir": str(root / group), "config": str(root / f"launch_{group}.yaml")})
    manifest = {"protocol_precedence": ["CRISP", "CAPI", "iBOT"],
                "checkpoint": config["checkpoint"], "checkpoint_key": config.get("checkpoint_key", "teacher"),
                "run_dir": str(root), "jobs": jobs,
                "submitted_at": datetime.now(timezone.utc).isoformat()}
    if dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "launch_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("This suite was already submitted; resume its Slurm jobs instead")
    for job in jobs:
        run = dict(config)
        run.update(output_dir=job["run_dir"], result_json=None, num_workers=4,
                   wandb_run_name=f"{config.get('wandb_run_name') or Path(config['checkpoint']).stem}-{job['group']}")
        run["evaluations"] = {name: name in job["evaluations"] for name in config["evaluations"]}
        Path(job["config"]).write_text(yaml.safe_dump(run, sort_keys=False))
        command = ["sbatch", "--parsable", f"--job-name=eval-crisp-{job['group']}",
                   f"--gpus-per-node={job['gpu_count']}", f"--cpus-per-task={job['cpus']}",
                   f"--mem={job['memory_gib']}G", f"--time={job['time_limit']}",
                   f"--output={root}/slurm_%j.out", f"--error={root}/slurm_%j.err",
                   str(REPO / "slurm/evaluation.sh"), job["config"]]
        job["job_id"] = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Submitted {job['group']}: {job['job_id']} ({job['gpu_count']} GPUs, {job['memory_gib']} GiB, {job['time_limit']})", flush=True)
    dependencies = ":".join(job["job_id"] for job in jobs)
    manifest["merge_job_id"] = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency=afterany:{dependencies}",
         f"--output={root}/merge_%j.out", f"--error={root}/merge_%j.err",
         str(REPO / "slurm/evaluation_merge.sh"), str(root)], cwd=REPO, text=True).strip().split(";")[0]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    from evaluation.merge_results import merge
    merge(root)
    print(f"Combined summary: {root / 'full_evaluation.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    submit(args.config, args.output_dir, args.dry_run)
