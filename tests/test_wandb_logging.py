import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from utils import wandb_logging


class WandbLoggingTest(unittest.TestCase):
    def args(self, mode='online', **overrides):
        values = wandb_logging.configure_wandb({'wandb_mode': mode, 'wandb_project': 'test-project', **overrides})
        return SimpleNamespace(**values, output_dir=Path('/tmp/test-output'))

    def test_online_loads_repo_key_without_putting_it_in_config_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = 'test-key-for-unit-tests-only'
            (root / '.wandb_key').write_text(key + '\n')
            sdk = SimpleNamespace(Settings=mock.Mock(), init=mock.Mock())
            args = self.args(wandb_entity='my-team', wandb_run_id='run-id', wandb_resume='allow')
            config = {'seed': 7, 'wandb_mode': 'online'}
            with mock.patch.object(wandb_logging, 'REPO_ROOT', root), mock.patch.dict('sys.modules', {'wandb': sdk}):
                with mock.patch.dict(os.environ, {'WANDB_API_KEY': 'unrelated-environment-key'}):
                    run = wandb_logging.init_wandb_run(args, config, 'evaluation')
                    self.assertEqual(os.environ['WANDB_API_KEY'], 'unrelated-environment-key')
            sdk.Settings.assert_called_once_with(api_key=key)
            call = sdk.init.call_args.kwargs
            self.assertEqual(call['mode'], 'online')
            self.assertEqual(call['entity'], 'my-team')
            self.assertEqual(call['id'], 'run-id')
            self.assertEqual(call['resume'], 'allow')
            self.assertIs(call['config'], config)
            self.assertNotIn(key, repr(config))
            self.assertNotIn(key, repr(vars(args)))
            self.assertIs(run, sdk.init.return_value)

    def test_missing_and_empty_online_key_fail_before_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = SimpleNamespace(Settings=mock.Mock(), init=mock.Mock())
            with mock.patch.object(wandb_logging, 'REPO_ROOT', root), mock.patch.dict('sys.modules', {'wandb': sdk}):
                with self.assertRaises(FileNotFoundError):
                    wandb_logging.init_wandb_run(self.args(), {}, 'training')
                (root / '.wandb_key').write_text('\n')
                with self.assertRaisesRegex(ValueError, 'one nonempty'):
                    wandb_logging.init_wandb_run(self.args(), {}, 'training')
            sdk.init.assert_not_called()

    def test_offline_and_disabled_do_not_require_a_key(self):
        sdk = SimpleNamespace(Settings=mock.Mock(), init=mock.Mock())
        with mock.patch.dict('sys.modules', {'wandb': sdk}), mock.patch.object(Path, 'read_text', side_effect=AssertionError('Key should not be read')):
            self.assertIsNone(wandb_logging.init_wandb_run(self.args('disabled'), {}, 'evaluation'))
            sdk.init.assert_not_called()
            wandb_logging.init_wandb_run(self.args('offline'), {}, 'training')
            self.assertEqual(sdk.init.call_args.kwargs['mode'], 'offline')
            sdk.Settings.assert_not_called()

    def test_yaml_resume_requires_id_and_has_explicit_default(self):
        with self.assertRaisesRegex(ValueError, 'requires wandb_run_id'):
            self.args(wandb_resume='must')
        self.assertEqual(self.args(wandb_run_id='id').wandb_resume, 'must')
        for settings in ({'wandb_mode': 'wrong'}, {'wandb_resume': True}, {'wandb_project': ''}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                wandb_logging.configure_wandb(settings)

    def test_evaluation_metrics_are_separate_per_task(self):
        run = SimpleNamespace(log=mock.Mock(), summary={})
        wandb_logging.log_evaluation(run, 'imagenet_knn_1pct', {'metrics': {'top1': 60, 'top5': 80, 'nested': {'x': 1}}})
        wandb_logging.log_evaluation(run, 'pascal_voc_1shot', {'metrics': {'map_percent': 50}})
        self.assertEqual(run.summary, {'imagenet_knn_1pct/top1': 60, 'imagenet_knn_1pct/top5': 80, 'pascal_voc_1shot/map_percent': 50})
        self.assertEqual(run.log.call_count, 2)


if __name__ == '__main__':
    unittest.main()
