import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evaluation.online_probes import (
    ONLINE_EVALUATIONS, OnlineProbeRunner, probe_due, probe_environment,
    run_probe_checkpoint, validate_probe_data,
)
from evaluation.utils.common import checkpoint_fingerprint, evaluation_identity, write_json


class OfflineProtocolTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / 'checkpoint.pth'
        self.checkpoint.write_bytes(b'checkpoint fixture')
        self.output = self.root / 'epoch0010.json'

    def worker(self, command, **kwargs):
        name = command[2].rsplit('.', 1)[1]
        options = dict(zip(command[4::2], command[5::2]))
        args = SimpleNamespace(
            checkpoint=Path(command[3]), checkpoint_key=options['--checkpoint-key'],
            arch=options['--arch'], seed=int(options['--seed']),
            datasets_root=Path(options['--datasets-root']),
            classification_manifests=Path(options['--classification-manifests']),
        )
        write_json(Path(options['--result-json']), {
            'status': 'completed', 'evaluation': name,
            'model': {'checkpoint_key': args.checkpoint_key,
                      'checkpoint_fingerprint': checkpoint_fingerprint(args.checkpoint)},
            'evaluation_identity': evaluation_identity(args),
            'metrics': {'top1': 42} if name == 'imagenet_knn' else {'miou': .42},
            'validation_sweep': [{'parameter': 123}],
        })
        return SimpleNamespace(returncode=0)

    def run_suite(self):
        return run_probe_checkpoint(self.checkpoint, self.root, self.output, epoch=10, seed=17)

    def test_offline_entrypoints_results_and_cache_validation(self):
        with mock.patch('evaluation.online_probes.subprocess.run', side_effect=self.worker) as launch:
            result = self.run_suite()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(list(result['evaluations']), list(ONLINE_EVALUATIONS))
        self.assertEqual(launch.call_count, 3)
        for call, name in zip(launch.call_args_list, ONLINE_EVALUATIONS):
            command = call.args[0]
            self.assertEqual(command[2], 'evaluation.utils.' + name)
            self.assertEqual(command[command.index('--checkpoint-key') + 1], 'teacher')
            self.assertEqual(command[command.index('--seed') + 1], '17')
            self.assertEqual('--batch-size' in command, name != 'imagenet_knn')
            self.assertNotIn('--k', command)
            self.assertEqual(result['evaluations'][name]['validation_sweep'], [{'parameter': 123}])
        with mock.patch('evaluation.online_probes.subprocess.run') as launch:
            self.assertEqual(self.run_suite()['status'], 'completed')
            launch.assert_not_called()
        self.checkpoint.write_bytes(b'changed checkpoint')
        with mock.patch('evaluation.online_probes.subprocess.run', side_effect=self.worker) as launch:
            self.run_suite()
            self.assertEqual(launch.call_count, 3)

    def test_failure_still_attempts_other_two_tasks(self):
        def fail_first(command, **kwargs):
            if command[2].endswith('pascal_voc_knn'):
                return SimpleNamespace(returncode=9)
            return self.worker(command, **kwargs)
        with mock.patch('evaluation.online_probes.subprocess.run', side_effect=fail_first) as launch:
            result = self.run_suite()
        self.assertEqual(launch.call_count, 3)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(list(result['errors']), ['pascal_voc_knn'])
        self.assertEqual(len(result['evaluations']), 2)

    def test_successful_exit_without_valid_results_is_failure(self):
        with mock.patch('evaluation.online_probes.subprocess.run', return_value=SimpleNamespace(returncode=0)):
            result = self.run_suite()
        self.assertEqual(len(result['errors']), 3)
        self.assertEqual(json.loads(self.output.read_text())['status'], 'failed')


class RunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'checkpoint.pth'
        self.source.write_bytes(b'epoch10')
        self.args = SimpleNamespace(
            online_probes_enabled=True, online_probe_frequency=10, output_dir=str(self.root),
            online_probe_datasets_root='dataset', arch='auto', seed=0,
            online_probe_batch_size=128, online_probe_num_workers=0,
            online_probe_max_concurrent_jobs=1, online_probe_gpu=None,
        )
        self.runner = OnlineProbeRunner(self.args)

    def test_disabled_and_non_due_epochs_do_nothing(self):
        with mock.patch('evaluation.online_probes.subprocess.Popen') as launch:
            self.assertIsNone(self.runner.submit(9, self.source))
            self.args.online_probes_enabled = False
            self.assertIsNone(self.runner.submit(10, self.source))
            launch.assert_not_called()
        self.assertFalse((self.root / 'online_probes').exists())

    def test_busy_worker_queues_immutable_snapshots_and_drains_at_exit(self):
        processes = []
        def launch(command, **kwargs):
            process = mock.Mock(pid=len(processes) + 1, returncode=0)
            process.poll.return_value = None
            def finish():
                process.poll.return_value = 0
                path = Path(command[command.index('--output') + 1])
                write_json(path, {'status': 'completed', 'evaluations': {
                    name: {'status': 'completed', 'metrics': {'score': 42}}
                    for name in ONLINE_EVALUATIONS
                }})
            process.wait.side_effect = finish
            processes.append(process)
            return process
        with mock.patch('evaluation.online_probes.subprocess.Popen', side_effect=launch) as popen:
            first = self.runner.submit(10, self.source)
            self.source.write_bytes(b'epoch20')
            second = self.runner.submit(20, self.source)
            self.assertIsNone(self.runner.submit(20, self.source))
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(self.runner.collect_completed(), [])
            self.assertEqual(first['checkpoint'].read_bytes(), b'epoch10')
            self.assertEqual(second['checkpoint'].read_bytes(), b'epoch20')
            self.runner.close(wait=True)
            self.assertEqual(popen.call_count, 2)
        records = self.runner.collect_completed()
        self.assertEqual([row['online_probe_epoch'] for row in records], [10, 20])
        for name in ONLINE_EVALUATIONS:
            self.assertEqual(records[0]['online_' + name + '_score'], 42)
        self.assertEqual(self.runner.collect_completed(), [])

    def test_dead_worker_reports_failure(self):
        process = mock.Mock(pid=12, returncode=-9)
        process.poll.return_value = -9
        path = self.root / 'epoch0010.json'
        self.runner.processes = [(process, mock.Mock())]
        self.runner.process_results[12] = (10, path)
        self.runner.submitted_results = [(10, path)]
        record, = self.runner.collect_completed()
        self.assertEqual(record['online_probe_success'], 0)
        self.assertIn('-9', json.loads(path.read_text())['error'])


class ConfigurationTest(unittest.TestCase):
    def test_interval(self):
        self.assertTrue(probe_due(15, 5))
        self.assertFalse(probe_due(14, 5))
        for frequency in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                probe_due(10, frequency)

    def test_training_yaml_controls_are_validated(self):
        import yaml
        from train import load_config
        path = Path('test-training.yaml')
        values = dict(arch='vit_small', register=0, additional_epochs=20, warmup_epochs=0,
                      data_path='dataset', initial_checkpoint='checkpoint.pth',
                      output_dir='output', wandb_mode='disabled')
        for enabled, frequency in ((False, 3), (True, 7)):
            values.update(online_probes_enabled=enabled, online_probe_frequency=frequency)
            with mock.patch.object(Path, 'open', mock.mock_open(read_data=yaml.safe_dump(values))):
                args = load_config(path)
            self.assertEqual(args.online_probes_enabled, enabled)
            self.assertEqual(args.online_probe_frequency, frequency)
        for key, value in (('online_probes_enabled', 'false'), ('online_probe_frequency', 0)):
            invalid = {**values, key: value}
            with mock.patch.object(Path, 'open', mock.mock_open(read_data=yaml.safe_dump(invalid))):
                with self.assertRaisesRegex(ValueError, key):
                    load_config(path)

    def test_worker_isolated_from_training_distributed_environment(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,5', 'RANK': '0',
                'LOCAL_RANK': '0', 'WORLD_SIZE': '8', 'MASTER_PORT': '12345',
                'TORCHELASTIC_RUN_ID': 'training'}, clear=True):
            environment = probe_environment()
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '3')
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID'):
                self.assertNotIn(key, environment)
            self.assertEqual(probe_environment(5)['CUDA_VISIBLE_DEVICES'], '5')
            with self.assertRaises(ValueError):
                probe_environment(7)

    def test_missing_data_fails_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                validate_probe_data(directory)

    def test_logging_uses_snapshot_epoch_and_existing_wandb_run(self):
        from train import log_online_probe_records
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'online_probes').mkdir()
            runner, writer, wandb = mock.Mock(), mock.Mock(), mock.Mock()
            runner.collect_completed.return_value = [
                {'online_probe_epoch': 10, 'online_pascal_voc_linear_miou': .42},
                {'online_probe_epoch': 20, 'online_probe_success': 0},
            ]
            log_online_probe_records(runner, directory, writer, wandb)
            self.assertEqual(wandb.log.call_count, 2)
            self.assertEqual(wandb.log.call_args.args[0]['train/online_probe_epoch'], 20)
            writer.add_scalar.assert_any_call('online_pascal_voc_linear_miou', .42, 10)
            self.assertEqual(len((Path(directory) / 'online_probes/metrics.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
