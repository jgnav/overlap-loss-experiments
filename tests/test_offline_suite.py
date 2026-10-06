"""Verify task allocations and combined result isolation without submitting jobs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evaluation.launch_slurm import submit
from evaluation.merge_results import merge


class OfflineSuiteTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "checkpoint.pth").write_bytes(b"fixture")
        (self.root / "datasets").mkdir()
        self.config = self.root / "eval.yaml"
        self.config.write_text(yaml.safe_dump({
            "checkpoint": "checkpoint.pth", "datasets_root": "datasets",
            "checkpoint_key": "teacher", "seed": 0,
            "evaluations": {"pascal_voc_knn": True, "imagenet_linear": True,
                            "pascal_voc_multilabel": True, "davis_vos": True,
                            "scannet_correspondence": False},
        }))

    def test_dry_run_does_not_create_output_and_preserves_requested_tasks(self):
        output = self.root / "suite"
        plan = submit(self.config, output, dry_run=True)
        self.assertFalse(output.exists())
        self.assertEqual([job["gpu_count"] for job in plan["jobs"]], [1, 4, 4, 1])
        selected = [name for job in plan["jobs"] for name in job["evaluations"]]
        self.assertEqual(set(selected), {"pascal_voc_knn", "imagenet_linear", "pascal_voc_multilabel", "davis_vos"})
        self.assertEqual(len(selected), len(set(selected)))

    def test_submission_uses_resource_overrides_and_merge_waits_for_all_groups(self):
        output = self.root / "suite"
        with mock.patch("evaluation.launch_slurm.subprocess.check_output", side_effect=["101\n", "102\n", "103\n", "104\n", "105\n"]) as slurm:
            plan = submit(self.config, output)
        commands = [call.args[0] for call in slurm.call_args_list]
        self.assertIn("--gpus-per-node=1", commands[0])
        self.assertIn("--gpus-per-node=4", commands[1])
        self.assertIn("--dependency=afterany:101:102:103:104", commands[-1])
        self.assertEqual(plan["merge_job_id"], "105")
        self.assertEqual(merge(output)["completed_evaluations"], [])
        for job in plan["jobs"]:
            config = yaml.safe_load(Path(job["config"]).read_text())
            self.assertEqual([name for name, value in config["evaluations"].items() if value], job["evaluations"])
            run = Path(job["run_dir"]); run.mkdir()
            (run / "full_evaluation.json").write_text(json.dumps({
                "checkpoint": plan["checkpoint"], "checkpoint_key": "teacher",
                "evaluation_identity": {"source": "same"}, "status": "completed",
                "evaluations": {name: {"metrics": {"score": 1}} for name in job["evaluations"]},
            }))
        combined = merge(output, final=True)
        self.assertEqual(combined["status"], "completed")
        self.assertEqual(len(combined["completed_evaluations"]), 4)
        # Missing group outputs cannot be reported as a successful full suite.
        (Path(plan["jobs"][0]["run_dir"]) / "full_evaluation.json").unlink()
        self.assertEqual(merge(output, final=True)["status"], "failed")
