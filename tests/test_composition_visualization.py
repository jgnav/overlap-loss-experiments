import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch

import composition_visualization as experiment


class CompositionMathTest(unittest.TestCase):
    def test_partition_is_complete_nonoverlapping_and_area_weighted(self):
        parent = (3, 7, 104, 80)
        for parts in (2, 4, 9):
            boxes, weights = experiment.partition_box(parent, parts)
            self.assertEqual(len(boxes), parts)
            self.assertAlmostEqual(weights.sum(), 1)
            areas = [(r - l) * (b - t) for l, t, r, b in boxes]
            self.assertEqual(sum(areas), (104 - 3) * (80 - 7))
            np.testing.assert_allclose(weights, np.asarray(areas) / sum(areas))

    def test_jensen_shannon_properties(self):
        p = np.array([.7, .2, .1])
        q = np.array([.1, .3, .6])
        self.assertAlmostEqual(experiment.jensen_shannon(p, p), 0)
        self.assertAlmostEqual(
            experiment.jensen_shannon(p, q), experiment.jensen_shannon(q, p)
        )
        self.assertLessEqual(experiment.jensen_shannon(p, q), np.log(2))
        self.assertAlmostEqual(experiment.jensen_shannon([1, 0], [0, 1]), np.log(2))

    def test_cross_image_donor_pairing_is_deterministic(self):
        regions = [
            experiment.CocoRegion(i, i, 1, "x", "unused", np.ones((2, 2), bool), (0, 0, 2, 2))
            for i in range(8)
        ]
        first = experiment.pair_shuffled_regions(regions, seed=4)
        second = experiment.pair_shuffled_regions(regions, seed=4)
        self.assertEqual([x.donor.image_id for x in first], [x.donor.image_id for x in second])
        self.assertTrue(all(x.target.image_id != x.donor.image_id for x in first))

    def test_bootstrap_clusters_multiple_parents_by_image(self):
        rows = []
        for model in ("a", "b"):
            for image_id in (1, 2, 3):
                for annotation_id in (1, 2):
                    for parts in (2, 4, 9):
                        value = image_id / 100 + parts / 1000
                        rows.append(experiment.Measurement(
                            model, image_id, annotation_id, 99, parts,
                            value, value + .1, .1,
                        ))
        summary = experiment.bootstrap_summary(rows, samples=100, seed=2)
        self.assertEqual(len(summary), 2 * 3 * 3)
        item = next(
            x for x in summary
            if x["model"] == "a" and x["parts"] == 4 and x["metric"] == "composition_js"
        )
        self.assertEqual(item["images"], 3)
        self.assertAlmostEqual(item["mean"], .024)


class CompositionIOTest(unittest.TestCase):
    def test_coco_loader_decodes_mask_and_uses_supported_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "coco" / "images" / "val2017"
            annotation_dir = root / "coco" / "annotations"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir()
            images, annotations = [], []
            for image_id in (1, 2):
                filename = f"{image_id:012d}.jpg"
                Image.new("RGB", (100, 100), "white").save(image_dir / filename)
                images.append({"id": image_id, "file_name": filename, "height": 100, "width": 100})
                annotations.append({
                    "id": image_id + 10, "image_id": image_id, "category_id": 1,
                    "area": 3600, "bbox": [20, 20, 60, 60], "iscrowd": 0,
                    "segmentation": [[20, 20, 80, 20, 80, 80, 20, 80]],
                })
            (annotation_dir / "instances_val2017.json").write_text(json.dumps({
                "images": images, "annotations": annotations,
                "categories": [{"id": 1, "name": "object"}],
            }))
            with (mock.patch.object(experiment, "MIN_PARENT_SIDE", 20),
                  mock.patch.object(experiment, "MIN_INSTANCE_AREA", .01)):
                regions = experiment.load_coco_regions(root, num_images=2, seed=0)
            self.assertEqual(len(regions), 2)
            self.assertTrue(all(region.mask.any() for region in regions))
            self.assertTrue(all(Path(region.image_path).is_file() for region in regions))

    def test_encode_region_is_one_sample_and_averages_patch_softmax(self):
        class Backbone(torch.nn.Module):
            num_register_tokens = 0
            def forward(self, tensor, return_all_tokens=True):
                self.batch_shape = tuple(tensor.shape)
                patches = torch.zeros((1, 4, 3), device=tensor.device)
                return torch.cat((torch.zeros((1, 1, 3), device=tensor.device), patches), dim=1)
        class Head(torch.nn.Module):
            def forward(self, tokens):
                logits = torch.tensor(
                    [[[2., 0.], [0., 2.], [2., 0.], [0., 2.]]], device=tokens.device
                )
                return tokens[:, 0], logits
        image = Image.new("RGB", (20, 20), "white")
        backbone = Backbone()
        with (mock.patch.object(experiment, "DEVICE", "cpu"),
              mock.patch.object(experiment, "INPUT_SIZE", 4),
              mock.patch.object(experiment, "TEMPERATURE", 1.0)):
            q = experiment.encode_region(image, (0, 0, 20, 20), backbone, Head(), 2)
        self.assertEqual(backbone.batch_shape[0], 1)
        np.testing.assert_allclose(q, [.5, .5], atol=1e-7)

    def test_evaluate_pair_uses_separate_calls_and_shuffled_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path, donor_path = root / "target.png", root / "donor.png"
            Image.new("RGB", (90, 90), "red").save(target_path)
            Image.new("RGB", (90, 90), "blue").save(donor_path)
            target = experiment.CocoRegion(
                1, 10, 1, "a", str(target_path), np.ones((90, 90), bool), (0, 0, 90, 90)
            )
            donor = experiment.CocoRegion(
                2, 20, 2, "b", str(donor_path), np.ones((90, 90), bool), (0, 0, 90, 90)
            )
            calls = []
            def fake_encode(image, box, backbone, head, patch_size):
                calls.append((image.getpixel((0, 0)), box))
                return np.array([.8, .2]) if image.getpixel((0, 0))[0] > 200 else np.array([.1, .9])
            with mock.patch.object(experiment, "encode_region", side_effect=fake_encode):
                rows = experiment.evaluate_pair(
                    experiment.RegionPair(target, donor), "model", None, None, 16
                )
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(calls), 1 + 2 * sum(experiment.PART_COUNTS))
            self.assertTrue(all(row.composition_js == 0 for row in rows))
            self.assertTrue(all(row.shuffled_js > 0 and row.specificity_gap > 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
