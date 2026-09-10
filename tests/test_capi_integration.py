import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from evaluation.utils import common, dense
from evaluation.vendor.capi import eval_segmentation as capi
from model.vision_transformer import VisionTransformer


class CAPIIntegrationTest(unittest.TestCase):
    def test_distributed_parameter_ties_preserve_single_rank_grid_order(self):
        classifier = capi.Classifier()
        classifier.hparam_grids = {'choice': (0, 1, 2)}
        classifier.ignore_labels = ()
        classifier.fit = mock.Mock()
        classifier.unfit = mock.Mock()
        classifier.predict = mock.Mock(return_value=torch.zeros(1))

        def gather(destination, value):
            destination[:] = [{0: 0.5, 2: 0.8}, {1: 0.8}]

        with mock.patch.object(torch.distributed, 'get_rank', return_value=0), \
             mock.patch.object(torch.distributed, 'get_world_size', return_value=2), \
             mock.patch.object(torch.distributed, 'barrier'), \
             mock.patch.object(torch.distributed, 'all_gather_object', side_effect=gather), \
             mock.patch.dict(capi.metrics_dict, {'mIoU': lambda *a: 0.8}):
            classifier.select_hparams(*(torch.zeros(1) for _ in range(4)))
        self.assertEqual(classifier.choice, 1)

    def test_patch_sizes_preserve_256_tokens_and_pixel_alignment(self):
        for ps in (14, 16):
            with self.subTest(patch_size=ps):
                resolution = dense.dense_resolution(ps)
                self.assertEqual(resolution, {14: 224, 16: 256}[ps])
                transform, target_transform = dense._dense_transforms(resolution)
                image = transform(Image.new('RGB', (350, 280)))
                target = np.arange(256, dtype=np.uint8).reshape(16, 16).repeat(ps, 0).repeat(ps, 1)
                labels = target_transform(Image.fromarray(target))
                model = VisionTransformer(img_size=[224], patch_size=ps, embed_dim=12, depth=1, num_heads=3).eval()
                with torch.no_grad():
                    tokens = model.get_intermediate_layers(image[None], n=1)[0][:, 1:]
                self.assertEqual(tokens.shape, (1, 256, 12))
                expected = torch.arange(256, dtype=torch.uint8)[:, None].expand(-1, ps * ps)
                self.assertTrue(torch.equal(dense._patchify_labels(labels[None], 16, 16), expected))

    def test_checkpoint_loading_infers_patch_size_and_position_grid(self):
        def factory(architecture, **kwargs):
            return VisionTransformer(embed_dim=12, depth=1, num_heads=3, **kwargs)
        for ps in (14, 16):
            with self.subTest(patch_size=ps), tempfile.TemporaryDirectory() as directory:
                source = factory('vit_small', img_size=[ps * 4], patch_size=ps)
                path = Path(directory) / 'checkpoint.pth'
                torch.save({'teacher': source.state_dict(), 'args': {'arch': 'vit_small'}}, path)
                with mock.patch.object(common, 'create_model', side_effect=factory):
                    loaded, metadata = common.load_backbone(path)
                self.assertEqual(metadata['patch_size'], ps)
                self.assertEqual(metadata['pretraining_position_grid'], 4)
                self.assertTrue(torch.equal(loaded.pos_embed, source.pos_embed))
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

                state = source.state_dict()
                state['blocks.0.ls1.gamma'] = torch.ones(12)
                torch.save({'teacher': state, 'args': {'arch': 'vit_small'}}, path)
                with mock.patch.object(common, 'create_model', side_effect=factory), \
                     self.assertRaisesRegex(ValueError, 'unsupported backbone'):
                    common.load_backbone(path)

    def test_unsupported_patch_size_fails(self):
        with self.assertRaises(ValueError):
            dense.dense_resolution(8)

    def test_upstream_knn_matches_brute_force_pixel_voting(self):
        torch.manual_seed(0)
        features = torch.randn(45, 3)
        labels = torch.randint(0, 3, (45, 4), dtype=torch.uint8)
        labels[0] = 255
        queries = torch.randn(3, 3)
        # Avoid CPU compiler startup in this small numerical regression test.
        original = capi.KNNClassifier._find_closest_chunk._torchdynamo_orig_callable
        with mock.patch.object(capi.KNNClassifier, '_find_closest_chunk', original):
            for distance in ('cosine', 'L2'):
                classifier = capi.KNNClassifier((255,), device='cpu', train_set_chunk_size=17)
                classifier.num_neighbors = 3
                classifier.distance = distance
                classifier.fit(features, labels)
                actual = classifier.predict(queries)
                keys = features[1:]
                distances = (torch.cdist(queries, keys) if distance == 'L2' else
                             1 - torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(keys, dim=-1).T)
                nearest = distances.topk(3, largest=False).indices
                expected = labels[1:][nearest].mode(dim=1).values
                self.assertTrue(torch.equal(actual, expected))

    def test_json_conversion_preserves_upstream_scores(self):
        for classifier, key, grid in (
            ('knn', 'knn', [{'num_neighbors': k, 'distance': d} for k in (1, 3, 10, 30) for d in ('cosine', 'L2')]),
            ('linear', 'logreg', [{'C': float(c), 'max_iter': 1000, 'tol': 1e-12, 'linesearch_max_iter': 50, 'lbfgs_hessian_rank': 5} for c in 10 ** np.linspace(-6, 5, 8)]),
        ):
            raw = {f'labels_{key}_mIoU': .45, f'labels_{key}_acc': .8}
            for i, params in enumerate(grid):
                suffix = '_'.join(f'{name}={value}' for name, value in params.items())
                raw[f'hparam_fitting.{key}.mIoU_{suffix}'] = i / 10
            result = dense._format_capi_result(raw, classifier)
            self.assertEqual(result['metrics']['miou_percent'], 45)
            self.assertEqual(result['metrics']['pixel_accuracy_percent'], 80)
            self.assertEqual(result['selected_hyperparameters'], grid[-1])

    def test_upstream_evaluation_holdout_refit_and_scoring_on_cpu(self):
        from torch.utils.data import TensorDataset
        torch.manual_seed(0)
        train = TensorDataset(torch.randn(40, 3), torch.randint(0, 3, (40, 4)))
        test = TensorDataset(torch.randn(5, 3), torch.randint(0, 3, (5, 4)))
        extracted = []

        def extract(model, dataset, *args, **kwargs):
            extracted.append(dataset)
            rows = [dataset[i] for i in range(len(dataset))]
            return (torch.stack([row[0] for row in rows])[:, None, None],
                    torch.stack([row[1] for row in rows])[:, None, None])

        def gather(destination, value):
            destination[0] = value

        original = capi.KNNClassifier._find_closest_chunk._torchdynamo_orig_callable
        with mock.patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
             mock.patch.object(capi, 'extract_features', side_effect=extract), \
             mock.patch.object(capi.KNNClassifier, '_find_closest_chunk', original), \
             mock.patch.object(torch.distributed, 'get_rank', return_value=0), \
             mock.patch.object(torch.distributed, 'get_world_size', return_value=1), \
             mock.patch.object(torch.distributed, 'broadcast'), \
             mock.patch.object(torch.distributed, 'barrier'), \
             mock.patch.object(torch.distributed, 'all_gather_object', side_effect=gather):
            np.random.seed(0)
            raw = capi.eval_model(
                torch.nn.Linear(3, 3), train_dataset_name=train, test_dataset_name=test,
                classifiers=('knn',), classifiers_kwargs={'knn': {'device': 'cpu'}},
                ignore_labels=(255,), resolution=256,
            )
        self.assertEqual([len(d) for d in extracted], [36, 5, 4])
        self.assertFalse(set(extracted[0].indices) & set(extracted[2].indices))
        result = dense._format_capi_result(raw, 'knn')
        self.assertEqual(len(result['validation_sweep']), 8)
        self.assertTrue(np.isfinite(result['metrics']['miou']))


if __name__ == '__main__':
    unittest.main()
