"""Check paper geometry, released propagation equivalence and GT isolation."""
import tempfile
import json
import zipfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from evaluation.utils import video_dinov3 as vos


class DINOv3VideoTest(unittest.TestCase):
    def test_geometry_matches_both_paper_footnote_examples(self):
        self.assertEqual(vos.resized_size((854, 480), 960, 16), (1712, 960))
        self.assertEqual(vos.resized_size((854, 480), 960, 14), (1708, 966))
        self.assertEqual(vos.resized_size((854, 480), 480, 16), (848, 480))
        self.assertEqual(vos.resized_size((480, 854), 480, 16), (480, 848))
        self.assertEqual(vos.RESOLUTIONS[16], {"small": 480, "medium": 960, "large": 1440})

    def test_chunked_affinity_matches_released_full_tensor_equations(self):
        generator = torch.Generator().manual_seed(29)
        current = F.normalize(torch.randn(2, 3, 8, generator=generator), dim=-1)
        features = F.normalize(torch.randn(3, 2, 3, 8, generator=generator), dim=-1)
        probabilities = torch.randn(3, 2, 3, 4, generator=generator).softmax(-1)
        # Equations of DINOv3's released propagate with unrestricted mask.
        dot = torch.einsum('ijd,tuvd->ijtuv', current, features).flatten(2).flatten(0, 1)
        cutoff = torch.topk(dot, dim=1, k=5).values[:, -1:]
        weights = torch.where(dot >= cutoff, dot, -torch.inf).div(.2).softmax(1)
        expected = weights @ probabilities.flatten(0, 2)
        expected /= expected.sum(1, keepdim=True)
        actual = vos.propagate(current.flatten(0, 1), list(features.flatten(1, 2)),
                               [p.flatten(0, 1).T for p in probabilities], chunk_size=2)
        torch.testing.assert_close(actual.T, expected, rtol=2e-6, atol=2e-7)

    def test_topk_includes_ties_and_has_no_local_restriction(self):
        current = torch.tensor([[1., 0.]])
        features = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])
        masks = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        actual = vos.propagate(current, [features], [masks], topk=1)
        torch.testing.assert_close(actual, torch.tensor([[.5], [.5]]))

    def test_selected_blocks_and_registers_not_removed_twice(self):
        from model.vision_transformer import VisionTransformer
        model = VisionTransformer(img_size=[32], patch_size=16, embed_dim=12,
                                  depth=4, num_heads=3, num_register_tokens=2).eval()
        image = torch.randn(3, 32, 32)
        with torch.no_grad():
            expected = F.normalize(model.get_intermediate_layers(image[None], n=1)[0][0, 1:], dim=-1)
            actual, grid = vos.patch_features(model, image, 16, feature_blocks=1)
            layers = model.get_intermediate_layers(image[None], n=4)
            expected_four = F.normalize(torch.stack([layer[0, 1:] for layer in layers]).mean(0), dim=-1)
            actual_four, grid_four = vos.patch_features(model, image, 16)
        self.assertEqual(grid, (2, 2))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(grid_four, (2, 2))
        torch.testing.assert_close(actual_four, expected_four)

    def test_first_frame_labels_align_and_unknown_ids_are_background(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 7
        mask[24:, 24:] = 255
        probs = vos.initial_probabilities(mask, (7,), (2, 2), 'cpu')
        torch.testing.assert_close(probs.sum(0), torch.ones(4))
        torch.testing.assert_close(probs[1], torch.tensor([1., 0., 0., 0.]))

    def test_later_object_never_receives_ground_truth_seed(self):
        class Backbone:
            def get_intermediate_layers(self, image, n=1):
                return [torch.cat((torch.zeros(1, 1, 4), torch.eye(4)[None]), dim=1) for _ in range(n)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for i in range(4):
                frame = root / f'{i:05d}.jpg'
                Image.new('RGB', (32, 32), 'gray').save(frame)
                frames.append(frame)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[:16, :16] = 1
                if i:
                    mask[16:, 16:] = 2
                Image.fromarray(mask).save(root / f'{i:05d}.png')
            with patch.object(vos, 'initial_probabilities', wraps=vos.initial_probabilities) as initialize:
                scores, details = vos.score_video(Backbone(), 16, frames, root, 32, 'cpu', 'davis')
            self.assertEqual(initialize.call_count, 1)
            self.assertEqual(list(scores), ['1'])
            self.assertEqual(details['first_frame_object_ids'], [1])
            self.assertEqual(details['scored_frames'], 2)

    def test_custom_datasets_require_explicit_manifests(self):
        for dataset in ('youtube_vos', 'mose'):
            with self.assertRaisesRegex(FileNotFoundError, 'explicit custom split'):
                vos.dataset_layout(Path('/tmp'), dataset)

    def test_protocol_dispatch_does_not_consume_dataset_manifest_as_json_abbreviation(self):
        from evaluation.utils.common import base_parser
        args, remaining = base_parser('dispatch').parse_known_args([
            'checkpoint.pth', '--video-protocol', 'dinov3',
            '--video-split-manifests-json', '{}', '--video-split-manifest', '/tmp/split.json'])
        self.assertEqual(args.video_split_manifests, {})
        self.assertEqual(remaining, ['--video-split-manifest', '/tmp/split.json'])

    def test_explicit_mosev2_split_is_supported_without_claiming_author_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'dataset': 'mose', 'dataset_release': 'v2', 'source': 'local holdout, seed0',
                        'author_split_verified': False, 'paths_relative_to': 'datasets_root',
                        'image_root': 'MOSEv2/train/JPEGImages', 'mask_root': 'MOSEv2/train/Annotations',
                        'splits': {'selection': ['a'], 'evaluation': ['b']}}
            path = root / 'split.json'
            path.write_text(json.dumps(manifest))
            images, _, names, details = vos.dataset_layout(root, 'mose', path)
            self.assertEqual(images, root / 'MOSEv2/train/JPEGImages')
            self.assertEqual(names, ['b'])
            self.assertFalse(details['author_split_verified'])
            manifest['author_split_verified'] = True
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                vos.dataset_layout(root, 'mose', path)
            manifest['author_split_verified'] = False
            manifest['splits']['evaluation'] = ['a']
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                vos.dataset_layout(root, 'mose', path)

    def test_zip_rgb_decoding_matches_disk_and_index_preserves_frame_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disk = root / 'image.png'
            Image.new('RGB', (854, 480), 'red').save(disk)
            archive = root / 'frames.zip'
            with zipfile.ZipFile(archive, 'w') as source:
                source.write(disk, 'RGB/video/00002.png')
                source.write(disk, 'RGB/video/00001.png')
            with zipfile.ZipFile(archive) as source:
                frames = vos._frames(zipfile.Path(source, 'RGB/video/'))
                self.assertEqual([p.stem for p in frames], ['00001', '00002'])
                actual, native = vos.read_image(frames[0], 480, 16)
                expected, expected_native = vos.read_image(disk, 480, 16)
                self.assertEqual(native, expected_native)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unannotated_youtube_frames_are_propagated_without_ground_truth_reseeding(self):
        class Backbone:
            def get_intermediate_layers(self, image, n=1):
                return [torch.cat((torch.zeros(1, 1, 4), torch.eye(4)[None]), dim=1) for _ in range(n)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for i in range(6):
                path = root / f'{i:05d}.jpg'
                Image.new('RGB', (32, 32), 'gray').save(path)
                frames.append(path)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[:16, :16] = 1
            Image.fromarray(mask).save(root / '00000.png')
            mask[16:, 16:] = 2
            Image.fromarray(mask).save(root / '00005.png')
            with patch.object(vos, 'propagate', wraps=vos.propagate) as propagate:
                scores, details = vos.score_video(Backbone(), 16, frames, root, 32, 'cpu', 'youtube_vos')
            self.assertEqual(propagate.call_count, 5)
            self.assertEqual(details['frames'], 6)
            self.assertEqual(details['scored_frames'], 1)
            self.assertEqual(list(scores), ['1'])

    def test_youtube_clip_begins_at_first_supplied_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / 'images', root / 'masks'
            images.mkdir(); masks.mkdir()
            for i in range(6):
                (images / f'{i:05d}.jpg').touch()
            (masks / '00002.png').touch()
            (masks / '00005.png').touch()
            frames, details = vos.evaluation_frames(images, masks, 'youtube_vos')
            self.assertEqual([p.stem for p in frames], ['00002', '00003', '00004', '00005'])
            self.assertEqual(details['skipped_leading_unannotated_frames'], 2)
            self.assertEqual(details['initialization_frame'], '00002')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
