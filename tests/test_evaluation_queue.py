import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
import evaluation_queue as queue


class QueueTests(unittest.TestCase):
    def make_runs(self, root):
        runs = []
        for name in ('first', 'second'):
            checkpoint = root / f'{name}.pth'
            checkpoint.touch()
            config = root / f'{name}.yaml'
            output = root / name
            config.write_text(yaml.safe_dump(dict(
                checkpoint=str(checkpoint), output_dir=str(output),
                evaluations={'task': True},
            )))
            runs.append(dict(checkpoint=str(checkpoint), config=str(config),
                             output_dir=str(output), log=str(root / f'{name}.log')))
        manifest = root / 'runs.json'
        manifest.write_text(json.dumps(runs))
        return manifest

    def test_sequential_retry_and_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_runs(root)
            (root / 'evaluation.py').write_text('''
import json, sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
root = Path.cwd()
name = Path(config['checkpoint']).stem
with (root / 'order.txt').open('a') as f:
    f.write(name + '\\n')
marker = root / (name + '.attempted')
if name == 'first' and not marker.exists():
    marker.touch()
    raise SystemExit(1)
output = Path(config['output_dir'])
output.mkdir(exist_ok=True)
(output / 'full_evaluation.json').write_text(json.dumps(dict(
    status='completed', checkpoint=config['checkpoint'], completed_evaluations=['task'])))
''')
            with patch.object(queue, 'ROOT', root):
                self.assertEqual(queue.main(['--runs', str(manifest), '--retry-delay', '1']), 0)
            self.assertEqual((root / 'order.txt').read_text().splitlines(), ['first', 'second', 'first'])
            state = json.loads((root / 'output/evaluation_queue/status.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual([r['attempts'] for r in state['runs']], [2, 1])
            self.assertTrue(all(r['pid'] is None for r in state['runs']))

    def test_rejects_duplicate_runs_and_incomplete_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_runs(root)
            runs = queue.load_runs(manifest)
            self.assertFalse(queue.result_completed(runs[0]))
            data = json.loads(manifest.read_text())
            data.append(data[0])
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                queue.load_runs(manifest)


if __name__ == '__main__':
    unittest.main()
