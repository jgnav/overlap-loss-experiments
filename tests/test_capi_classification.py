"""Exercise CAPI head training, split isolation, padding, selection and resume."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

from evaluation.utils import capi_classification as adapter
from evaluation.vendor.capi import eval_classification as capi
from evaluation.vendor.capi import classification_support as support
from model.vision_transformer import VisionTransformer


class CAPIClassificationTest(unittest.TestCase):
    def test_adapter_matches_final_normalized_features_and_omits_registers(self):
        model = VisionTransformer(img_size=[32], patch_size=16, embed_dim=64,
                                  depth=2, num_heads=4, num_register_tokens=2).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            tokens = model.get_intermediate_layers(images, n=1)[0]
            cls, registers, patches = adapter.CAPIBackbone(model)(images)
        torch.testing.assert_close(cls, tokens[:, 0])
        torch.testing.assert_close(patches.flatten(1, 2), tokens[:, 1:])
        self.assertEqual(registers.shape, (2, 0, 64))

    def test_train_and_holdout_transforms_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'class'; p.mkdir()
            Image.new('RGB', (8, 8)).save(p / 'image.jpg')
            original = ImageFolder(directory)
            train = adapter.make_dataset(original, transform=lambda _: 'train')
            val = adapter.make_dataset(original, transform=lambda _: 'holdout')
            self.assertEqual(train[0][0], 'train')
            self.assertEqual(val[0][0], 'holdout')
            self.assertIsNone(original.transform)

    def test_padded_samples_are_excluded_from_metric(self):
        metric = capi.AnyMatchAccuracy()
        predictions = torch.tensor([[[9., 0.], [0., 9.], [0., 0.]]])
        targets = torch.tensor([[[0], [1]]])
        metric.update(predictions, targets)
        torch.testing.assert_close(metric.compute(), torch.tensor(1.0))
        dataset = support.DatasetWithEnumeratedTargets(TensorDataset(torch.zeros(3, 1), torch.arange(3)),
                                                      pad_dataset=True, num_replicas=4)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset[3][1][0], -1)

    def test_crisp_epoch_budget_and_scaled_fixed_learning_rate(self):
        for count in (1, 2, 4, 8):
            p = adapter.protocol(count, training_samples=1153050)
            self.assertEqual(p['global_batch_size'], p['batch_size_per_gpu'] * count)
            self.assertEqual(p['epochs'], 200)
            self.assertEqual(p['iterations'], 225206)
            self.assertEqual(p['warmup_iterations'], 1250)
            self.assertEqual(p['optimizer'], 'AdamW')
            self.assertEqual(p['learning_rates'], [.001])
            self.assertEqual(p['actual_initial_learning_rates'], [.004])
            self.assertEqual(len(p['learning_rates']) * len(p['weight_decays']), 3)
        with self.assertRaises(ValueError):
            adapter.protocol(3)

    def test_official_training_selection_and_checkpoint_resume_on_cpu(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(64))
            def forward(self, x):
                features = x.mean(dim=(1, 2, 3))[:, None] * self.weight[None]
                return features, None, features[:, None, None].expand(-1, 2, 2, -1)

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory) / 'store'
            dist.init_process_group('gloo', init_method=rendezvous.as_uri(), rank=0, world_size=1)
            try:
                torch.manual_seed(0)
                class IntegerTargets(TensorDataset):
                    def __getitem__(self, index):
                        image, target = super().__getitem__(index)
                        return image, int(target)
                train = IntegerTargets(torch.randn(40, 3, 8, 8), torch.arange(40) % 3)
                test = IntegerTargets(torch.randn(5, 3, 8, 8), torch.arange(5) % 3)
                observed = []
                def dataset(dataset_str_or_path, **kwargs):
                    observed.append(kwargs.get('transform'))
                    return dataset_str_or_path
                output = Path(directory) / 'evaluation'; output.mkdir()
                model = Backbone()
                with mock.patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
                     mock.patch.object(torch.nn.Module, 'cuda', lambda self, *a, **kw: self), \
                     mock.patch.object(capi, 'make_dataset', side_effect=dataset), \
                     mock.patch.object(support.SmoothedValue, 'synchronize_between_processes'):
                    def run(steps):
                        return capi.eval_model(model, lambda _: None, output_dir=str(output),
                                              train_dataset_name=train, test_dataset_names=('test',),
                                              n_iters=steps, warmup_iters=1, save_checkpoint_period=1,
                                              eval_period=0, batch_size=2, num_classes=3,
                                              num_workers=0, use_compile=False,
                                              learning_rates=(.001,), weight_decays=(.001,))
                    # Resolve the test identifier without changing result-key behavior.
                    def local_dataset(dataset_str_or_path, **kwargs):
                        observed.append(kwargs.get('transform'))
                        return test if dataset_str_or_path == 'test' else dataset_str_or_path
                    with mock.patch.object(capi, 'make_dataset', side_effect=local_dataset):
                        run(2)
                        checkpoint = output / 'checkpoints/last_checkpoint.pth'
                        saved = torch.load(checkpoint, weights_only=False)
                        self.assertEqual(saved['iteration'], 1)
                        self.assertEqual(len(list(checkpoint.parent.glob('model_*.pth'))), 1)
                        self.assertFalse(model.weight.requires_grad)
                        rows = json.loads((output / 'test_classifiers.json').read_text())
                        self.assertEqual({r['feature_source'] for r in rows}, {'cls', 'avg_patch', 'cls_avg_patch', 'patch'})
                        # A completed-step resume performs no additional training.
                        with mock.patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('unexpected training')):
                            run(2)
                        self.assertFalse(any(p.grad is not None for p in model.parameters()))
                self.assertIsNot(observed[0], observed[1])
            finally:
                dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
