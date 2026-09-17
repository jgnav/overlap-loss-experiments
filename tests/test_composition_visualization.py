import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch

import composition_visualization as viz


class CompositionMathTest(unittest.TestCase):
    def test_coco_discovery_polygon_rle_and_ambiguity(self):
        from pycocotools import mask as mask_utils
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'coco' / 'train2017'
            annotations = root / 'coco' / 'annotations'
            images.mkdir(parents=True)
            annotations.mkdir()
            Image.new('RGB', (16, 8), 'white').save(images / '000000240684.jpg')
            b = np.zeros((8, 16), dtype=np.uint8)
            b[:, 6:] = 1
            rle = mask_utils.encode(np.asfortranarray(b))
            rle['counts'] = rle['counts'].decode('ascii')
            data = {'images': [{'id': 240684, 'file_name': '000000240684.jpg', 'height': 8, 'width': 16}],
                    'categories': [{'id': 1, 'name': 'dog'}, {'id': 2, 'name': 'car'}],
                    'annotations': [
                        {'id': 10, 'image_id': 240684, 'category_id': 1, 'area': 100,
                         'segmentation': [[0, 0, 8, 0, 8, 8, 0, 8]]},
                        {'id': 11, 'image_id': 240684, 'category_id': 2, 'area': 80,
                         'segmentation': rle}]}
            source = annotations / 'instances_train2017.json'
            source.write_text(json.dumps(data))
            with (mock.patch.object(viz, 'OUTPUT_DIR', root / 'out'),
                  mock.patch.object(viz, 'CONCEPT_A', 'Dog'),
                  mock.patch.object(viz, 'CONCEPT_B', 'Car')):
                image, mask, record = viz.find_coco_pair(240684, root)
                self.assertEqual(image.name, '000000240684.jpg')
                self.assertEqual(record['objects'][0]['annotation_id'], 10)
                self.assertEqual(record['ambiguous_pixels_excluded'], 16)
                pixels = np.asarray(Image.open(mask))
                np.testing.assert_array_equal(pixels[0, 0], viz.OBJECT_A_COLOR)
                np.testing.assert_array_equal(pixels[0, 10], viz.OBJECT_B_COLOR)
                np.testing.assert_array_equal(pixels[0, 6], viz.BACKGROUND_COLOR)
                # The same mask in uncompressed COCO RLE form.
                flat = b.flatten(order='F')
                counts, previous, count = [], 0, 0
                for value in flat:
                    if value != previous:
                        counts.append(count)
                        count, previous = 0, value
                    count += 1
                counts.append(count)
                data['annotations'][1]['segmentation'] = {'size': [8, 16], 'counts': counts}
                source.write_text(json.dumps(data))
                _, mask, _ = viz.find_coco_pair(240684, root)
                np.testing.assert_array_equal(np.asarray(Image.open(mask)), pixels)
                with self.assertRaises(FileNotFoundError):
                    viz.find_coco_pair(99, root)
                (annotations / 'instances_duplicate.json').write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'multiple annotation'):
                    viz.find_coco_pair(240684, root)

    def test_contrast_sets_are_disjoint_and_ties_are_other(self):
        a, b = viz.associated_sets(np.array([.5, .1, .2, .2]), np.array([.1, .5, .2, .2]), 3)
        np.testing.assert_array_equal(a, [0])
        np.testing.assert_array_equal(b, [1])
        with self.assertRaisesRegex(ValueError, 'contrasting'):
            viz.associated_sets(np.ones(4) / 4, np.ones(4) / 4)

    def test_component_masses_partition_full_distribution(self):
        x = np.array([[.5, .1, .2, .2], [.1, .3, .1, .5]])
        masses = viz.component_mass(x, np.array([0]), np.array([1]))
        np.testing.assert_allclose(masses, [[.5, .1, .4], [.1, .3, .6]])
        np.testing.assert_allclose(masses.sum(1), 1)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            viz.component_mass(x, [0], [0])

    def test_exact_mixture_counts_no_replacement_and_union_identity(self):
        rng = np.random.default_rng(17)
        x = rng.dirichlet(np.ones(5), 20)
        ia, ib = np.arange(12), np.arange(12, 20)
        mu_a, mu_b = np.array([.5, .1, .1, .1, .2]), np.array([.1, .5, .1, .1, .2])
        result = viz.build_composition(x, ia, ib, mu_a, mu_b, 24, max_mixture_patches=7)
        self.assertEqual(result.mixture_counts, [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)])
        np.testing.assert_allclose(result.means[2], .6 * result.means[0] + .4 * result.means[1])
        for indices, counts, mean in zip(result.mixture_indices, result.mixture_counts, result.mixture_means):
            self.assertEqual(len(np.unique(indices)), 4)
            self.assertEqual(np.count_nonzero(indices < 12), counts[0])
            np.testing.assert_allclose(x[indices].mean(0), mean)
        repeated = viz.build_composition(x, ia, ib, mu_a, mu_b, 24, max_mixture_patches=7)
        np.testing.assert_array_equal(result.mixture_means, repeated.mixture_means)
        self.assertTrue(np.isnan(result.patch_masses[20:]).all())

    def test_only_probability_modes_are_used(self):
        logits = torch.tensor([[.2, .1, -.1], [-.2, .4, .1]])
        np.testing.assert_allclose(viz.normalize_bank(logits, 'softmax', .2), (logits / .2).softmax(-1).numpy())
        np.testing.assert_allclose(viz.normalize_bank(logits, 'sinkhorn', .2).sum(1), 1, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            viz.normalize_bank(logits, 'raw_logits', .2)


class CompositionIOTest(unittest.TestCase):
    def test_teacher_head_loading_uses_raw_logits_and_checkpoint_region_mode(self):
        from model import iBOTHead
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            for shared in (True, False):
                head = iBOTHead(12, 5, patch_out_dim=5, shared_head=shared).eval()
                checkpoint = {
                    'teacher': {'head.' + k: v for k, v in head.state_dict().items()},
                    'args': {'shared_head': shared, 'out_dim': 5, 'patch_out_dim': 5,
                             'region_normalization': 'sinkhorn', 'region_temp': .23},
                    # Deliberately no centers: the region branch must not need them.
                }
                torch.save(checkpoint, path)
                backbone = torch.nn.Identity()
                backbone.embed_dim = 12
                with mock.patch.object(viz, 'load_backbone', return_value=(backbone, {'patch_size': 16})):
                    _, loaded, _, dimensions, mode, temperature = viz.load_teacher(path)
                self.assertEqual((dimensions, mode, temperature), (5, 'sinkhorn', .23))
                inputs = torch.randn(1, 5, 12)
                torch.testing.assert_close(loaded(inputs)[1], head(inputs)[1])
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_palette_masks_resize_together_and_patch_purity_is_area_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (20, 12), 'white').save(root / 'image.png')
            labels = np.zeros((12, 20), dtype=np.uint8)
            labels[:, :10] = 1
            mask = Image.fromarray(labels, mode='P')
            mask.putpalette([0, 0, 0, 255, 0, 0] + [0] * (768 - 6))
            mask.save(root / 'mask.png')
            image, resized, geometry = viz.prepare_pair(root / 'image.png', root / 'mask.png', 4, 20)
            self.assertEqual(image.size, (20, 12))
            selected, fractions = viz.select_patches(resized, (255, 0, 0), 4, .9)
            np.testing.assert_allclose(fractions.reshape(3, 5)[0], [1, 1, .5, 0, 0])
            self.assertEqual(selected.sum(), 6)
            _, padded, geometry = viz.prepare_pair(root / 'image.png', root / 'mask.png', 8, 20)
            self.assertEqual(padded.shape, (16, 24, 3))
            self.assertTrue((padded[12:] == 0).all())

    def test_calibration_cannot_reuse_displayed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.png'
            Image.new('RGB', (16, 16), 'red').save(image)
            copy = root / 'copy.png'
            Image.open(image).save(copy)
            checkpoint = root / 'checkpoint.pth'
            checkpoint.touch()
            refs = [viz.ReferenceRegion(copy, image, (255, 0, 0))]
            with mock.patch.object(viz, 'REFERENCE_A', refs), mock.patch.object(viz, 'REFERENCE_B', refs):
                with self.assertRaisesRegex(ValueError, 'leakage'):
                    viz.validate_inputs(checkpoint, image, image)

    def test_missing_calibration_fails_before_model_load(self):
        with mock.patch.object(viz, 'REFERENCE_A', []), mock.patch.object(viz, 'REFERENCE_B', []):
            with self.assertRaisesRegex(ValueError, 'REFERENCE_A'):
                viz.validate_inputs('missing', 'missing', 'missing')

    def test_main_pipeline_outputs_png_full_vectors_and_protocol_on_cpu(self):
        class Backbone(torch.nn.Module):
            num_register_tokens = 0
            def forward(self, x, return_all_tokens=True):
                # Tiny deterministic CPU tokenization, deliberately no pretrained claim.
                patches = torch.nn.functional.avg_pool2d(x, 4).flatten(2).transpose(1, 2)
                return torch.cat((patches.mean(1, keepdim=True), patches), dim=1)
        class Head(torch.nn.Module):
            def forward(self, x):
                return x[:, 0], x[:, 1:]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'synthetic.pth'
            checkpoint.touch()
            def pair(name, pixels, mask):
                image_path, mask_path = root / f'{name}.png', root / f'{name}_mask.png'
                Image.fromarray(pixels).save(image_path)
                Image.fromarray(mask).save(mask_path)
                return image_path, mask_path
            pixels = np.zeros((16, 32, 3), dtype=np.uint8)
            pixels[:, :16] = (230, 40, 20)
            pixels[:, 16:] = (20, 220, 40)
            mask = np.zeros_like(pixels)
            mask[:, :16] = (255, 0, 0)
            mask[:, 16:] = (0, 255, 0)
            target, target_mask = pair('mixed', pixels, mask)
            a, am = pair('ref_a', np.full((16, 16, 3), (200, 30, 20), dtype=np.uint8), np.full((16, 16, 3), (255, 0, 0), dtype=np.uint8))
            b, bm = pair('ref_b', np.full((16, 16, 3), (20, 190, 30), dtype=np.uint8), np.full((16, 16, 3), (0, 255, 0), dtype=np.uint8))
            with (mock.patch.object(viz, 'REFERENCE_A', [viz.ReferenceRegion(a, am, (255, 0, 0))]),
                  mock.patch.object(viz, 'REFERENCE_B', [viz.ReferenceRegion(b, bm, (0, 255, 0))]),
                  mock.patch.object(viz, 'OBJECT_A_COLOR', (255, 0, 0)),
                  mock.patch.object(viz, 'OBJECT_B_COLOR', (0, 255, 0)),
                  mock.patch.object(viz, 'LONG_SIDE', 32), mock.patch.object(viz, 'DPI', 65),
                  mock.patch.object(viz, 'OUTPUT_DIR', root),
                  mock.patch.object(viz, 'load_teacher', return_value=(Backbone(), Head(), {'patch_size': 4}, 3, 'softmax', .5)),
                  mock.patch.object(viz, 'find_coco_pair', return_value=(target, target_mask, {'image_id': 240684})),
                  mock.patch.object(viz, 'CHECKPOINT', checkpoint),
                  mock.patch.object(viz, 'COCO_IMAGE_ID', 240684),
                  mock.patch.object(viz, 'DATASETS_ROOT', root),
                  contextlib.redirect_stdout(io.StringIO())):
                viz.main()
            path = root / 'synthetic_mixed_composition.png'
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                self.assertGreater(image.width, 1000)
            record = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(record['reference_counts'], [1, 1])
            self.assertFalse(record['teacher_center_applied'])
            with np.load(path.with_suffix('.npz')) as arrays:
                self.assertEqual(arrays['region_means'].shape, (3, 3))
                self.assertEqual(arrays['mixture_means'].shape, (5, 3))
                np.testing.assert_allclose(arrays['region_means'].sum(1), 1, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
