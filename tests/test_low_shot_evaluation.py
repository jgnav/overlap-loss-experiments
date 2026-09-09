import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from evaluation.utils import classification, imagenet, orchestrator
from evaluation.utils.classification_data import (
    MultilabelDataset, few_shot_dataset, sample_few_shot_indices,
)


class FewShotTest(unittest.TestCase):
    def setUp(self):
        self.targets = np.zeros((12, 3), dtype=np.float32)
        self.targets[:8, 0] = 1
        self.targets[2:10, 1] = 1
        self.targets[4:, 2] = 1
        self.targets[:2, 2] = -1

    def test_draws_are_positive_unique_nested_and_reproducible(self):
        previous, previous_draws = set(), [[], [], []]
        for shots in (1, 2, 5):
            indices, draws = sample_few_shot_indices(self.targets, shots, 17)
            self.assertEqual((indices, draws), sample_few_shot_indices(self.targets, shots, 17))
            self.assertEqual(indices, sorted(set(indices)))
            self.assertTrue(previous <= set(indices))
            for column, draw in enumerate(draws):
                self.assertEqual(len(set(draw)), shots)
                self.assertTrue(np.all(self.targets[draw, column] == 1))
                self.assertEqual(draw[:len(previous_draws[column])], previous_draws[column])
            self.assertEqual(set(indices), set().union(*map(set, draws)))
            previous, previous_draws = set(indices), draws
        self.assertNotEqual(
            sample_few_shot_indices(self.targets, 2, 17),
            sample_few_shot_indices(self.targets, 2, 18),
        )

    def test_shared_images_are_not_duplicated_and_all_labels_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            images = [Path(directory) / f'{index}.jpg' for index in range(2)]
            for image in images:
                Image.new('RGB', (4, 4)).save(image)
            targets = np.asarray([[1, 1, -1], [0, 0, 1]], dtype=np.float32)
            full = MultilabelDataset(list(zip(images, targets)), ['cat', 'dog', 'bird'])
            subset, metadata = few_shot_dataset(full, 1, 0)
            self.assertEqual(subset.indices, [0, 1])
            self.assertEqual(len(subset), 2)
            self.assertEqual(subset.classes, full.classes)
            torch.testing.assert_close(subset[0][1], full.targets[0])
            self.assertEqual(metadata['drawn_indices_by_class'], {'cat': [0], 'dog': [0], 'bird': [1]})
            self.assertEqual(metadata['positive_counts_by_class'], {'cat': 1, 'dog': 1, 'bird': 1})
            self.assertEqual(metadata['selected_images'], list(map(str, images)))
            self.assertIn('selected_indices_sha256', json.loads(json.dumps(metadata)))

    def test_insufficient_positive_examples_fail_without_using_unknowns(self):
        with self.assertRaisesRegex(ValueError, 'at least 2 positive'):
            sample_few_shot_indices([[1, -1], [-1, 1], [0, 0]], 2, 0)
        with self.assertRaisesRegex(ValueError, '1-, 2-, or 5-shot'):
            sample_few_shot_indices(self.targets, 3, 0)

    def test_small_probe_can_train_with_missing_negative_labels(self):
        # A low-shot union may have no negatives for a class; do not apply the
        # full-manifest positive/negative coverage requirement to that union.
        logits = torch.zeros(2, 3, requires_grad=True)
        labels = torch.tensor([[1., 1., -1.], [1., 0., 1.]])
        loss = classification.classification_loss(logits, labels, multilabel=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.grad[0, 2].item(), 0)
        self.assertTrue((logits.grad[:, 0] < 0).all())


class ImageNetFractionsTest(unittest.TestCase):
    def test_exact_fraction_and_class_proportions(self):
        targets = [0] * 103 + [1] * 197 + [2] * 200
        for fraction in (0.01, 0.1, 1.0):
            selected = imagenet._stratified_subset_indices(targets, fraction, 7)
            self.assertEqual(len(selected), round(len(targets) * fraction))
            self.assertEqual(len(set(selected)), len(selected))
            for label in (0, 1, 2):
                actual = sum(targets[i] == label for i in selected)
                self.assertLessEqual(abs(actual - targets.count(label) * fraction), 1)
            self.assertEqual(selected, imagenet._stratified_subset_indices(targets, fraction, 7))

    def test_all_regimes_report_correct_bank_and_full_validation(self):
        vocabulary = {str(i): i for i in range(1000)}
        class Folder:
            class_to_idx = vocabulary
            classes = list(vocabulary)
            targets = [i for i in range(1000) for _ in range(100)]
            def __len__(self):
                return len(self.targets)
        full_train = Folder()
        # These mocks replace image/GPU I/O and neighbor computation, while
        # exercising the real bank selection, feature normalization and report.
        class Validation:
            class_to_idx = vocabulary
            def __len__(self):
                return 2000
        validation = Validation()
        seen = []
        def extract(model, dataset, args, description):
            seen.append(len(dataset))
            return torch.ones(len(dataset), 2), torch.zeros(len(dataset), dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(checkpoint=Path(directory) / 'model.pth', checkpoint_key='teacher',
                                   arch='vit_small', datasets_root=Path(directory), seed=7,
                                   result_json=Path(directory) / 'result.json')
            for name, fraction in imagenet.IMAGENET_KNN_FRACTIONS.items():
                with self.subTest(evaluation=name), contextlib.ExitStack() as stack:
                    seen.clear()
                    for target, value in (
                        ('load_backbone', (mock.MagicMock(), {})),
                        ('_resolve_imagenet_root', Path(directory)),
                        ('datasets.ImageFolder', full_train),
                        ('IndexedImageFolder', validation),
                        ('is_main_process', True),
                        ('evaluation_identity', {}),
                        ('_weighted_knn', {'top1': 42., 'top5': 60.}),
                    ):
                        stack.enter_context(mock.patch.object(imagenet, target, return_value=value) if '.' not in target
                                            else mock.patch('evaluation.utils.imagenet.' + target, return_value=value))
                    stack.enter_context(mock.patch.object(imagenet, '_extract_distributed_features', side_effect=extract))
                    stack.enter_context(mock.patch.object(imagenet.dist, 'barrier'))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    result = imagenet.run_imagenet_knn(args, name)
                    self.assertEqual(seen, [round(100000 * fraction), 2000])
                    self.assertEqual(result['evaluation'], name)
                    self.assertEqual(result['protocol']['training_fraction'], fraction)
                    self.assertEqual(result['metrics'], result['metrics_by_neighbors']['20'])
                    self.assertEqual(json.loads(args.result_json.read_text()), result)


class ReportingTest(unittest.TestCase):
    def test_every_fraction_and_shot_count_has_a_distinct_row(self):
        results = {
            name: {'metrics': {'top1': value}}
            for name, value in zip(imagenet.IMAGENET_KNN_FRACTIONS, (60, 70, 80))
        }
        results['imagenet_linear'] = {'metrics': {'top1': 81}}
        for shots in (1, 2, 5):
            results[f'pascal_voc_{shots}shot'] = {'dataset': 'PASCAL VOC', 'metrics': {'map_percent': shots * 10}}
        results['pascal_voc_multilabel'] = {'dataset': 'PASCAL VOC', 'metrics': {'map_percent': 90}}
        rows = orchestrator._result_table(results)
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['dataset'] for r in rows if 'knn' in r},
                         {'ImageNet-1K 1%', 'ImageNet-1K 10%', 'ImageNet-1K 100%'})
        self.assertEqual({r['regime'] for r in rows if r['dataset'] == 'PASCAL VOC'},
                         {'1shot', '2shot', '5shot', 'full'})

    def test_shot_only_selection_preflights_voc_manifest_and_positive_supply(self):
        args = SimpleNamespace(datasets_root=Path('/unused'), seed=0)
        chosen = [entry for entry in orchestrator.EVALUATIONS if entry[0] == 'pascal_voc_5shot']
        samples = {'train': [(Path('image'), [1, 0]), (Path('image2'), [0, 1])]}
        with mock.patch('evaluation.utils.classification_data.read_multilabel_manifest', return_value=(samples, [], {})) as reader:
            with self.assertRaisesRegex(ValueError, 'at least 5 positive'):
                orchestrator._preflight_classification(args, chosen)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(reader.call_args.args[2], 'pascal_voc')


if __name__ == '__main__':
    unittest.main()
