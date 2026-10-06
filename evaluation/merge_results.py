"""Collect results from the resource-specific jobs of one offline suite."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def merge(directory, final=False):
    root = Path(directory).resolve()
    manifest = json.loads((root / "launch_manifest.json").read_text())
    results, groups, selected, identities = {}, {}, [], []
    for job in manifest["jobs"]:
        selected.extend(job["evaluations"])
        path = Path(job["run_dir"]) / "full_evaluation.json"
        if not path.is_file():
            groups[job["group"]] = "missing" if final else "pending"
            continue
        summary = json.loads(path.read_text())
        if summary["checkpoint"] != manifest["checkpoint"] or summary["checkpoint_key"] != manifest["checkpoint_key"]:
            raise ValueError(f"Mismatched checkpoint in {path}")
        groups[job["group"]] = summary["status"]
        identity = summary.get("evaluation_identity")
        if identity is not None:
            identities.append(identity)
        for name, result in summary.get("evaluations", {}).items():
            if name not in job["evaluations"] or name in results:
                raise ValueError(f"Unexpected or duplicated evaluation: {name}")
            results[name] = result
    if identities and any(value != identities[0] for value in identities):
        raise ValueError("Task groups used different evaluation sources or datasets")
    complete = set(results) == set(selected) and all(state == "completed" for state in groups.values())
    status = "completed" if complete else ("failed" if final or "failed" in groups.values() else "running")
    summary = {"status": status, "checkpoint": manifest["checkpoint"],
               "checkpoint_key": manifest["checkpoint_key"], "selected_evaluations": selected,
               "completed_evaluations": list(results), "evaluations": results,
               "group_status": groups, "jobs": manifest["jobs"],
               "protocol_precedence": manifest["protocol_precedence"],
               "updated_at": datetime.now(timezone.utc).isoformat()}
    destination = root / "full_evaluation.json"
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(destination)
    print(f"Suite {status}: {len(results)}/{len(selected)} completed")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = merge(args.directory, args.final)
    raise SystemExit(1 if result["status"] == "failed" else 0)
