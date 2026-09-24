"""Evaluate immutable teacher snapshots using the unchanged offline protocols."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from evaluation.utils.common import REPO_ROOT, utc_now, write_json
from evaluation.utils.datasets import make_pascal_voc
from evaluation.utils.imagenet import _resolve_imagenet_root
from evaluation.utils.orchestrator import EVALUATIONS, _load_completed_result, evaluation_command
from evaluation.utils.runtime import worker_environment


ONLINE_FREQUENCY = 10
ONLINE_EVALUATIONS = ("pascal_voc_knn", "pascal_voc_linear", "imagenet_knn")


def probe_due(epoch, frequency=ONLINE_FREQUENCY):
    """Use one-based completed training epochs for the probe interval."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if type(frequency) is not int or frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    return epoch % frequency == 0


def validate_probe_data(datasets_root):
    """Validate the same ImageNet and original VOC splits used offline."""
    root = Path(os.path.expandvars(str(datasets_root))).expanduser().resolve()
    imagenet = _resolve_imagenet_root(root)
    classes = [{p.name for p in (imagenet / split).iterdir() if p.is_dir()}
               for split in ("train", "val")]
    if len(classes[0]) != 1000 or classes[0] != classes[1]:
        raise ValueError("ImageNet train/val must share exactly the same 1,000 classes")
    for split in ("train", "val"):
        voc = make_pascal_voc(root, split)
        for path in (*voc.images, *voc.targets):
            if not path.is_file():
                raise FileNotFoundError(f"VOC {split} file missing: {path}")
    return root


def probe_environment(gpu=None):
    """Isolate evaluators from torchrun and limit them to one allocated GPU."""
    environment = worker_environment()
    for key in list(environment):
        if key.startswith("TORCHELASTIC_") or key in {
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        }:
            environment.pop(key, None)
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    allocated = [item.strip() for item in visible.split(",") if item.strip()] if visible else []
    if gpu not in (None, "", "none", "None"):
        selected = str(gpu)
        if "," in selected or (visible is not None and selected not in allocated):
            raise ValueError("online_probe_gpu must select one GPU from CUDA_VISIBLE_DEVICES")
    else:
        selected = allocated[0] if allocated else ("" if visible == "" else "0")
    environment["CUDA_VISIBLE_DEVICES"] = selected
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return environment


def run_probe_checkpoint(checkpoint, datasets_root, output, arch="auto", seed=0,
                         batch_size=128, num_workers=0, epoch=None,
                         frequency=ONLINE_FREQUENCY):
    """Run all three offline entrypoints; preserve their complete result JSONs.

    Only feature-extraction resources are configurable here. Dataset fractions,
    transforms, classifiers, sweeps and scoring belong to the offline modules.
    A failed task does not prevent the other two tasks from being attempted.
    """
    if type(frequency) is not int or frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    datasets_root = Path(os.path.expandvars(str(datasets_root))).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    task_root = output.parent / output.stem
    task_root.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        checkpoint=checkpoint, checkpoint_key="teacher", arch=arch,
        datasets_root=datasets_root, output_dir=task_root, seed=seed,
        classification_manifests=datasets_root / "evaluation_manifests",
        segmentation_batch_size=batch_size, num_workers=num_workers,
    )
    modules = {name: module for name, module, _ in EVALUATIONS}
    environment = probe_environment()
    started, timer = utc_now(), time.monotonic()
    results, errors = {}, {}
    for name in ONLINE_EVALUATIONS:
        path = task_root / f"{name}.json"
        try:
            result = _load_completed_result(path, args, name)
            if result is None:
                print(f"Online epoch {epoch}: starting offline protocol {name}", flush=True)
                completed = subprocess.run(
                    evaluation_command(args, name, modules[name], path),
                    cwd=REPO_ROOT, env=environment, check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"{name} exited with status {completed.returncode}")
                result = _load_completed_result(path, args, name)
                if result is None:
                    raise RuntimeError(f"{name} did not produce a compatible completed result JSON")
            results[name] = result
        except (OSError, RuntimeError, ValueError) as error:
            errors[name] = f"{type(error).__name__}: {error}"
            print(f"Online epoch {epoch}: {errors[name]}", flush=True)
    result = {
        "evaluation": "online_probes", "protocol_version": 2,
        "status": "failed" if errors else "completed", "epoch": epoch,
        "checkpoint": str(checkpoint), "checkpoint_key": "teacher",
        "datasets_root": str(datasets_root), "seed": seed, "frequency": frequency,
        "started_at": started, "finished_at": utc_now(),
        "elapsed_seconds": time.monotonic() - timer,
        "selected_evaluations": list(ONLINE_EVALUATIONS),
        "evaluations": results, "errors": errors,
    }
    if errors:
        result["error"] = "; ".join(errors.values())
    write_json(output, result)
    return result


def immutable_checkpoint_copy(source, destination):
    """Copy the full checkpoint through a same-directory atomic rename."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


class OnlineProbeRunner:
    """Queue each scheduled epoch and bound the number of active workers."""

    def __init__(self, args, repository_root=None):
        self.args = args
        self.repository_root = Path(repository_root or REPO_ROOT)
        self.processes = []
        self.pending = []
        self.submitted_results = []
        self.completed_results = set()
        self.process_results = {}
        self.submitted_epochs = set()
        root = Path(args.output_dir) / "online_probes"
        reported = {}
        metrics_path = root / "metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    reported[record["online_probe_epoch"]] = record["online_probe_success"]
                except (ValueError, KeyError, TypeError):
                    continue
        for path in sorted(root.glob("epoch[0-9][0-9][0-9][0-9].json")):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                epoch = int(result["epoch"])
                status = result["status"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if status not in {"completed", "failed"}:
                continue
            self.submitted_results.append((epoch, path))
            self.submitted_epochs.add(epoch)
            if reported.get(epoch) == int(status == "completed"):
                self.completed_results.add(path)

    def _reap(self):
        active = []
        for process, log in self.processes:
            if process.poll() is None:
                active.append((process, log))
                continue
            log.close()
            epoch, path = self.process_results.pop(process.pid)
            try:
                terminal = json.loads(path.read_text()).get("status") in {"completed", "failed"}
            except (OSError, ValueError):
                terminal = False
            if not terminal:
                write_json(path, {"status": "failed", "epoch": epoch,
                                  "error": f"Worker exited with code {process.returncode} without a result"})
        self.processes = active

    def _launch_pending(self):
        maximum = self.args.online_probe_max_concurrent_jobs
        while self.pending and len(self.processes) < maximum:
            epoch, snapshot, result = self.pending.pop(0)
            log_path = result.parent / "logs" / f"epoch{epoch:04d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "evaluation.online_probes",
                "--checkpoint", str(snapshot), "--datasets-root",
                str(Path(os.path.expandvars(str(self.args.online_probe_datasets_root))).expanduser().resolve()),
                "--output", str(result), "--arch", self.args.arch, "--seed", str(self.args.seed),
                "--batch-size", str(self.args.online_probe_batch_size),
                "--num-workers", str(self.args.online_probe_num_workers),
                "--epoch", str(epoch), "--frequency", str(self.args.online_probe_frequency),
            ]
            log = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command, cwd=self.repository_root,
                    env=probe_environment(getattr(self.args, "online_probe_gpu", None)),
                    stdout=log, stderr=subprocess.STDOUT,
                )
            except (OSError, ValueError) as error:
                log.close()
                write_json(result, {"status": "failed", "epoch": epoch, "error": str(error)})
                continue
            self.processes.append((process, log))
            self.process_results[process.pid] = (epoch, result)
            print(f"Online probes: launched all three offline evaluations for epoch {epoch}", flush=True)

    def submit(self, epoch, checkpoint):
        if not getattr(self.args, "online_probes_enabled", False):
            return None
        if not probe_due(epoch, self.args.online_probe_frequency) or epoch in self.submitted_epochs:
            return None
        self._reap()
        root = Path(self.args.output_dir).resolve() / "online_probes"
        snapshot = immutable_checkpoint_copy(checkpoint, root / "checkpoints" / f"teacher_epoch{epoch:04d}.pth")
        result = root / f"epoch{epoch:04d}.json"
        write_json(result, {"status": "queued", "epoch": epoch, "checkpoint": str(snapshot),
                            "selected_evaluations": list(ONLINE_EVALUATIONS)})
        self.pending.append((epoch, snapshot, result))
        self.submitted_results.append((epoch, result))
        self.submitted_epochs.add(epoch)
        self._launch_pending()
        return {"epoch": epoch, "checkpoint": snapshot, "result": result}

    def collect_completed(self):
        self._reap()
        self._launch_pending()
        records = []
        for epoch, path in self.submitted_results:
            if path in self.completed_results:
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if result.get("status") not in {"completed", "failed"}:
                continue
            record = {"online_probe_epoch": epoch,
                      "online_probe_success": int(result["status"] == "completed")}
            if result["status"] == "failed":
                print(f"Online probes: FAILED epoch {epoch}: {result.get('error', 'unknown error')}", flush=True)
            for name in ONLINE_EVALUATIONS:
                task = result.get("evaluations", {}).get(name)
                record[f"online_{name}_success"] = int(task is not None and task.get("status") == "completed")
                if task:
                    for metric, value in task.get("metrics", {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            record[f"online_{name}_{metric}"] = float(value)
            records.append(record)
            self.completed_results.add(path)
        return records

    def retry_failed(self):
        """Retry failed saved snapshots after a training restart."""
        root = Path(self.args.output_dir) / "online_probes"
        for epoch, path in self.submitted_results:
            try:
                status = json.loads(path.read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError):
                continue
            snapshot = root / "checkpoints" / f"teacher_epoch{epoch:04d}.pth"
            if status == "failed" and snapshot.is_file():
                previous_log = root / "logs" / f"epoch{epoch:04d}.log"
                if previous_log.is_file():
                    shutil.copy2(previous_log, previous_log.with_suffix(".failed.log"))
                write_json(path, {"status": "queued", "epoch": epoch,
                                  "checkpoint": str(snapshot),
                                  "selected_evaluations": list(ONLINE_EVALUATIONS)})
                self.completed_results.discard(path)
                self.pending.append((epoch, snapshot, path))
        self._launch_pending()

    def close(self, wait=True):
        """Drain scheduled evaluations at normal exit, unless explicitly disabled.

        With wait=False, queued snapshots/reports remain available for manual
        retries. Running workers are left alone; Slurm may end them with the job.
        """
        self._reap()
        self._launch_pending()
        if wait:
            while self.processes or self.pending:
                for process, _ in self.processes:
                    process.wait()
                self._reap()
                self._launch_pending()
        else:
            for _, log in self.processes:
                log.close()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="auto", choices=("auto", "vit_small", "vit_base", "vit_large"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128, help="Segmentation feature-extraction batch size only")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--frequency", type=int, default=ONLINE_FREQUENCY)
    return parser


def main():
    args = _parser().parse_args()
    try:
        result = run_probe_checkpoint(**vars(args))
    except Exception as error:
        write_json(args.output, {"evaluation": "online_probes", "status": "failed",
                                "epoch": args.epoch, "error": f"{type(error).__name__}: {error}"})
        raise
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
