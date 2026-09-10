import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from evaluation.utils.common import checkpoint_fingerprint, evaluation_identity
from evaluation.utils.config import load_config
from evaluation.utils.runtime import worker_environment


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluation_entrypoint', ROOT / 'evaluation.py')
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


class EvaluationConfigTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'model.pth').write_bytes(b'checkpoint fixture')
        (self.root / 'data').mkdir()
        self.path = self.root / 'run.yaml'
        self.values = {
            'checkpoint': 'model.pth', 'datasets_root': 'data', 'output_dir': 'results',
            'checkpoint_key': 'student', 'arch': 'vit_small', 'seed': 17, 'num_workers': 3,
            'segmentation_batch_size': 9,
            'evaluations': {'imagenet_knn_1pct': True, 'pascal_voc_knn': True, 'coco_multilabel': False},
        }
        self.save()

    def save(self):
        self.path.write_text(yaml.safe_dump(self.values))

    def test_resolves_paths_from_config_and_disables_omitted_tasks(self):
        config = load_config(self.path)
        self.assertEqual(config.checkpoint, self.root / 'model.pth')
        self.assertEqual(config.datasets_root, self.root / 'data')
        self.assertEqual(config.output_dir, self.root / 'results')
        self.assertEqual(config.evaluations, ['pascal_voc_knn', 'imagenet_knn_1pct'])
        self.assertEqual(config.seed, 17)

    def test_invalid_configs_fail_before_launch(self):
        invalid = [
            {'evalutions': {}}, {'num_workers': True}, {'seed': -1}, {'seed': 2**32},
            {'segmentation_batch_size': 0}, {'checkpoint_key': 'invalid'}, {'arch': 'invalid'},
            {'evaluations': {}}, {'evaluations': {'imagenet_knn': False}},
            {'evaluations': {'imagenet_kn': True}}, {'evaluations': {'imagenet_knn': 'false'}},
            {'evaluations': {'imagenet_knn': 1}}, {'evaluations': ['imagenet_knn']},
            {'checkpoint': None}, {'output_dir': 123},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.path.write_text(yaml.safe_dump({**self.values, **changes}))
                with mock.patch.object(entrypoint.subprocess, 'run') as launch:
                    with self.assertRaises(ValueError):
                        entrypoint.main([str(self.path)])
                    launch.assert_not_called()

    def test_duplicate_keys_are_rejected(self):
        self.path.write_text('checkpoint: a\ncheckpoint: b\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_config(self.path)
        self.path.write_text('evaluations:\n  imagenet_knn: true\n  imagenet_knn: false\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_config(self.path)

    def test_worker_environment_discovers_cuda_wheels_without_modifying_parent(self):
        library = self.root / 'site-packages/nvidia/cublas/lib'
        library.mkdir(parents=True)
        with mock.patch('evaluation.utils.runtime.sysconfig.get_path', return_value=str(self.root / 'site-packages')):
            with mock.patch.dict(os.environ, {'LD_LIBRARY_PATH': '/existing'}):
                env = worker_environment()
                self.assertEqual(env['LD_LIBRARY_PATH'], f'{library}:/existing')
                self.assertEqual(os.environ['LD_LIBRARY_PATH'], '/existing')
                self.assertEqual(env['PYTHONUNBUFFERED'], '1')

    def worker_result(self, command, **kwargs):
        name = command[2].rsplit('.', 1)[-1]
        result_path = Path(command[command.index('--result-json') + 1])
        config = load_config(self.path)
        result_path.write_text(json.dumps({
            'status': 'completed', 'evaluation': name,
            'model': {'checkpoint_fingerprint': checkpoint_fingerprint(config.checkpoint),
                      'checkpoint_key': config.checkpoint_key},
            'evaluation_identity': evaluation_identity(config), 'metrics': {'top1': 60.0},
        }))
        return SimpleNamespace(returncode=0)

    def test_main_launches_only_enabled_tasks_and_resumes_saved_config(self):
        with mock.patch.object(entrypoint.subprocess, 'run', side_effect=self.worker_result) as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main([str(self.path)]), 0)
            self.assertEqual(launch.call_count, 2)
            dense, imagenet = [call.args[0] for call in launch.call_args_list]
            for command in (dense, imagenet):
                self.assertEqual(command[command.index('--checkpoint-key') + 1], 'student')
                self.assertEqual(command[command.index('--seed') + 1], '17')
                self.assertEqual(command[command.index('--num-workers') + 1], '3')
            self.assertEqual(dense[dense.index('--batch-size') + 1], '9')
            self.assertNotIn('--batch-size', imagenet)
            snapshot = self.root / 'results/evaluation_config.yaml'
            summary_path = self.root / 'results/full_evaluation.json'
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(set(summary['evaluations']), {'pascal_voc_knn', 'imagenet_knn_1pct'})
            self.assertEqual(load_config(snapshot).evaluations, load_config(self.path).evaluations)
            launch.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main([str(snapshot)]), 0)
            launch.assert_not_called()
            # A selection-only change also keeps matching completed results.
            self.values['evaluations'] = {'imagenet_knn_1pct': True}
            self.save()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main([str(self.path)]), 0)
            launch.assert_not_called()
            self.assertEqual(set(json.loads(summary_path.read_text())['evaluations']), {'imagenet_knn_1pct'})

    def test_failure_stops_following_tasks_and_records_failure(self):
        with mock.patch.object(entrypoint.subprocess, 'run', return_value=SimpleNamespace(returncode=7)) as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main([str(self.path)]), 1)
            self.assertEqual(launch.call_count, 1)
        summary = json.loads((self.root / 'results/full_evaluation.json').read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertIn('status 7', summary['error'])

    def test_successful_process_without_result_fails(self):
        with mock.patch.object(entrypoint.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main([str(self.path)]), 1)
        summary = json.loads((self.root / 'results/full_evaluation.json').read_text())
        self.assertIn('compatible completed result', summary['error'])

    def test_wandb_records_results_and_finishes_on_success_and_failure(self):
        self.values.update(wandb_mode='online', wandb_project='evaluation-test')
        self.save()
        for failed in (False, True):
            with self.subTest(failed=failed):
                self.values['output_dir'] = f'results-{failed}'
                self.save()
                run = SimpleNamespace(summary={}, log=mock.Mock(), finish=mock.Mock())
                effect = (lambda *a, **k: SimpleNamespace(returncode=3)) if failed else self.worker_result
                with mock.patch.object(entrypoint, 'init_wandb_run', return_value=run) as initialize:
                    with mock.patch.object(entrypoint.subprocess, 'run', side_effect=effect), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(entrypoint.main([str(self.path)]), int(failed))
                initialize.assert_called_once()
                self.assertEqual(initialize.call_args.args[0].wandb_project, 'evaluation-test')
                run.finish.assert_called_once_with(exit_code=int(failed))
                self.assertEqual(run.summary['state/status'], 'failed' if failed else 'completed')
                if not failed:
                    self.assertIn('imagenet_knn_1pct/top1', run.summary)


if __name__ == '__main__':
    unittest.main()
