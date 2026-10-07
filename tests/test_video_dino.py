"""Compare the 480p/four-block adapter with the actual pinned DINO code."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from evaluation.utils import video_dino as vos
from evaluation.utils.common import base_parser
from evaluation.utils.config import load_config
from evaluation.utils.orchestrator import evaluation_command


ROOT = Path(__file__).resolve().parents[1]


def upstream_functions():
    tree = ast.parse((ROOT / 'evaluation/vendor/dino/eval_video_segmentation.py').read_text())
    names = {'read_frame', 'read_seg', 'color_normalize', 'to_one_hot',
             'restrict_neighborhood', 'label_propagation', 'norm_mask', 'extract_feature'}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'cv2': cv2, 'np': np, 'torch': torch, 'F': F, 'Image': Image}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned DINO>', 'exec'), namespace)
    return namespace


class DINOVideoTest(unittest.TestCase):
    def test_rgb_and_initial_mask_match_upstream_for_both_geometries(self):
        original = upstream_functions()
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as directory:
            rgb, mask = Path(directory) / 'rgb.png', Path(directory) / 'mask.png'
            Image.fromarray(rng.integers(0, 256, (41, 73, 3), dtype=np.uint8)).save(rgb)
            labels = np.zeros((41, 73), dtype=np.uint8)
            labels[10:30, 20:60] = 3  # Preserve empty object-ID channels.
            Image.fromarray(labels).save(mask)
            for preserve_aspect, scale in ((True, [480]), (False, [480, 480])):
                expected, h, w = original['read_frame'](str(rgb), scale_size=scale)
                actual, native = vos.read_image(rgb, preserve_aspect)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(native, (w, h))
                probs, _ = original['read_seg'](str(mask), 16, scale_size=scale)
                grid = tuple(side // 16 for side in actual.shape[-2:])
                actual_probs = vos.initial_probabilities(labels, grid, 'cpu')
                torch.testing.assert_close(actual_probs, probs[0].flatten(1), rtol=0, atol=0)
            self.assertEqual(vos.resized_size((854, 480)), (832, 480))
            self.assertEqual(vos.resized_size((480, 854)), (480, 832))

    def test_propagation_matches_actual_dino_for_local_and_global_masks(self):
        original = upstream_functions()
        generator = torch.Generator().manual_seed(83)
        grid = (3, 4)
        target = torch.randn(12, 6, generator=generator)
        features = [torch.randn(12, 6, generator=generator) for _ in range(3)]
        probs = [torch.randn(3, 12, generator=generator).softmax(0) for _ in features]
        original['extract_feature'] = lambda *args, **kwargs: (target, *grid)
        for radius in (0, 1, 12):
            args = SimpleNamespace(size_mask_neighborhood=radius, topk=5)
            original['args'] = args
            # Run original CUDA-only functions on CPU, leaving their math intact.
            with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
                expected, _, _ = original['label_propagation'](
                    args, None, None, [value.T for value in features],
                    [value.reshape(1, 3, *grid) for value in probs])
            actual = vos.propagate(target, features, probs, grid, radius=radius, chunk_size=2)
            torch.testing.assert_close(actual, expected[0].flatten(1), rtol=2e-6, atol=3e-7)

    def test_postprocessing_matches_upstream_and_keeps_soft_history(self):
        original = upstream_functions()
        probs = torch.rand(3, 6, generator=torch.Generator().manual_seed(19))
        before = probs.clone()
        values = F.interpolate(probs.reshape(1, 3, 2, 3), scale_factor=16, mode='bilinear',
                               align_corners=False, recompute_scale_factor=False)[0]
        expected = original['norm_mask'](values).argmax(0).byte().numpy()
        expected = np.asarray(Image.fromarray(expected).resize((73, 41), Image.Resampling.NEAREST))
        actual = vos.prediction(probs, (2, 3), (73, 41), 16)
        np.testing.assert_array_equal(actual, expected)
        torch.testing.assert_close(probs, before, rtol=0, atol=0)
        # Constant positive channels yield NaNs in upstream norm_mask. Check
        # argmax equivalence rather than silently introducing a new policy.
        constant = torch.tensor([[1.] * 6, [0.] * 6])
        values = F.interpolate(constant.reshape(1, 2, 2, 3), scale_factor=16, mode='bilinear', align_corners=False)[0]
        expected = original['norm_mask'](values).argmax(0).byte().numpy()
        expected = np.asarray(Image.fromarray(expected).resize((73, 41), Image.Resampling.NEAREST))
        np.testing.assert_array_equal(vos.prediction(constant, (2, 3), (73, 41), 16), expected)

    def test_four_block_average_precedes_cosine_normalization(self):
        class Backbone:
            def get_intermediate_layers(self, image, n):
                self.n = n
                return [torch.tensor([[[100., 100.], [float(i), 2.]]]) for i in range(1, 5)]
        model = Backbone()
        features, grid = vos.patch_features(model, torch.zeros(3, 16, 16), 16)
        self.assertEqual(model.n, 4)
        self.assertEqual(grid, (1, 1))
        torch.testing.assert_close(features, torch.tensor([[2.5, 2.]]))

    def test_final_block_features_match_upstream(self):
        class Backbone:
            patch_embed = SimpleNamespace(patch_size=16)

            def get_intermediate_layers(self, image, n):
                self.n = n
                return [torch.tensor([[[100., 100.], [3., 2.], [4., 5.]]])]

        model = Backbone()
        image = torch.zeros(3, 16, 32)
        original = upstream_functions()
        with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
            expected, h, w = original['extract_feature'](model, image, return_h_w=True)
        actual, grid = vos.patch_features(model, image, 16, feature_blocks=1)
        self.assertEqual(model.n, 1)
        self.assertEqual(grid, (h, w))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dino_v1_configuration_accepts_both_feature_counts(self):
        original = (ROOT / 'config/evaluation_video_dino.yaml').read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            for blocks in (1, 4):
                path.write_text(original.replace('dino_480p_last4', 'dino_v1_480p')
                                .replace('video_feature_blocks: 4', f'video_feature_blocks: {blocks}'))
                args = load_config(path)
                command = evaluation_command(args, 'davis_vos', 'evaluation.utils.davis_vos', Path('/tmp/result.json'))
                worker = base_parser('test').parse_args(command[3:])
                self.assertEqual(worker.video_protocol, 'dino_v1_480p')
                self.assertEqual(worker.video_feature_blocks, blocks)

    def test_supplied_initialization_is_separate_from_scoring_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks, initial = root / 'gt', root / 'initial'
            masks.mkdir(); initial.mkdir()
            seed = np.zeros((32, 32), dtype=np.uint8)
            seed[:, :16] = 1
            full = seed.copy(); full[:, 16:] = 2
            frames = [root / f'{i:05d}.jpg' for i in range(3)]
            for frame in frames:
                Image.fromarray(full).save(masks / f'{frame.stem}.png')
            Image.fromarray(seed).save(initial / '00000.png')
            features = torch.ones(4, 2)
            with patch.object(vos, 'read_image', return_value=(torch.zeros(3, 32, 32), (32, 32))), \
                 patch.object(vos, 'patch_features', return_value=(features, (2, 2))), \
                 patch.object(vos, 'prediction', return_value=seed), \
                 patch.object(vos, 'propagate', side_effect=lambda target, feats, probs, grid: probs[0]):
                objects, details = vos.score_video(None, 16, frames, masks, 'cpu', None,
                                                  dataset='youtube_vos', feature_blocks=1,
                                                  initial_mask_folder=initial)
            self.assertEqual(set(objects), {'1'})
            self.assertEqual(details['object_ids'], [1])
            self.assertEqual(details['scored_frames'], 2)
            self.assertEqual(objects['1'], {'j': 1.0, 'f': 1.0})

    def test_clip_starts_at_supplied_initialization_not_full_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb, gt, seed = root / 'rgb' / 'clip', root / 'gt' / 'clip', root / 'seed' / 'clip'
            for folder in (rgb, gt, seed):
                folder.mkdir(parents=True)
            for index in range(3):
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(rgb / f'{index:05d}.jpg')
                Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(gt / f'{index:05d}.png')
            Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(seed / '00001.png')
            manifest = root / 'split.json'
            manifest.write_text(json.dumps({'dataset': 'youtube_vos', 'dataset_release': '2018',
                                           'author_split_verified': False, 'source': 'test',
                                           'paths_relative_to': 'datasets_root', 'image_root': 'rgb',
                                           'mask_root': 'gt', 'initial_mask_root': 'seed',
                                           'splits': {'selection': [], 'evaluation': ['clip']}}))
            images, _, names, split = vos.preflight_masks(root, 'youtube_vos', manifest)
            self.assertEqual(names, ['clip'])
            frames, details = vos.evaluation_frames(images / 'clip', Path(split['initial_mask_root']) / 'clip', 'youtube_vos')
            self.assertEqual([frame.stem for frame in frames], ['00001', '00002'])
            self.assertEqual(details['skipped_leading_unannotated_frames'], 1)

    def test_default_protocol_and_explicit_dinov3_remain_distinct(self):
        args = load_config(ROOT / 'config/evaluation_video_dino.yaml')
        self.assertEqual(args.video_protocol, 'dino_480p_last4')
        command = evaluation_command(args, 'davis_vos', 'evaluation.utils.davis_vos', Path('/tmp/result.json'))
        worker = base_parser('test').parse_args(command[3:])
        self.assertEqual(worker.video_protocol, args.video_protocol)
        self.assertEqual(worker.video_feature_blocks, 4)
        self.assertEqual(worker.video_resolution, 'small')
        self.assertEqual(load_config(ROOT / 'config/evaluation_video_dinov3.yaml').video_protocol, 'dinov3')
        with self.assertRaisesRegex(FileNotFoundError, 'explicit custom split manifest'):
            vos.preflight_masks('/tmp', 'youtube_vos')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
