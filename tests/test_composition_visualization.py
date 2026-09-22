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
    def test_partitions_are_complete_and_nonoverlapping(self):
        parent = (3, 7, 104, 80)
        parent_area = (104 - 3) * (80 - 7)
        for parts in (2, 4, 9):
            boxes = experiment.partition_box(parent, parts)
            self.assertEqual(len(boxes), parts)
            areas = [(right - left) * (bottom - top) for left, top, right, bottom in boxes]
            self.assertEqual(sum(areas), parent_area)

    def test_jensen_shannon_is_symmetric_bounded_and_zero_on_identity(self):
        p = np.array([.7, .2, .1])
        q = np.array([.1, .3, .6])
        self.assertAlmostEqual(experiment.jensen_shannon(p, p), 0)
        self.assertAlmostEqual(
            experiment.jensen_shannon(p, q), experiment.jensen_shannon(q, p)
        )
        self.assertLessEqual(experiment.jensen_shannon(p, q), np.log(2))
        self.assertAlmostEqual(experiment.jensen_shannon([1, 0], [0, 1]), np.log(2))

    def test_bootstrap_reports_object_mean_and_image_cluster_intervals(self):
        rows = []
        for model in ("control", "ours"):
            for image_id, values in ((1, (.1, .2)), (2, (.3,))):
                for annotation_offset, value in enumerate(values):
                    for parts in (2, 4, 9):
                        rows.append(experiment.Measurement(
                            model, image_id, 10 * image_id + annotation_offset,
                            1, parts, value + parts / 100,
                        ))
        summary = experiment.bootstrap_summary(rows, samples=100, seed=4)
        self.assertEqual(len(summary), 2 * 3)
        item = next(
            row for row in summary
            if row["model"] == "control" and row["parts"] == 4
        )
        self.assertEqual((item["objects"], item["images"]), (3, 2))
        self.assertAlmostEqual(item["mean_js"], (.14 + .24 + .34) / 3)


class CompositionIOTest(unittest.TestCase):
    def test_coco_loader_decodes_masks_and_derives_parent_boxes(self):
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
                images.append({
                    "id": image_id, "file_name": filename,
                    "height": 100, "width": 100,
                })
                annotations.append({
                    "id": image_id + 10, "image_id": image_id, "category_id": 1,
                    "area": 900, "bbox": [30, 30, 30, 30], "iscrowd": 0,
                    "segmentation": [[30, 30, 60, 30, 60, 60, 30, 60]],
                })
            (annotation_dir / "instances_val2017.json").write_text(json.dumps({
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "object"}],
            }))
            with (
                mock.patch.object(experiment, "MIN_PARENT_SIDE", 20),
                mock.patch.object(experiment, "MIN_INSTANCE_AREA", .01),
            ):
                regions = experiment.load_coco_regions(root, num_objects=2, seed=0)
            self.assertEqual(len(regions), 2)
            self.assertTrue(all(Path(region.image_path).is_file() for region in regions))
            self.assertTrue(all(region.box[0] < 30 and region.box[2] > 60 for region in regions))

    def test_encode_region_uses_one_forward_and_averages_patch_softmax(self):
        class Backbone(torch.nn.Module):
            num_register_tokens = 0

            def forward(self, tensor, return_all_tokens=True):
                self.batch_shape = tuple(tensor.shape)
                patches = torch.zeros((1, 4, 3), device=tensor.device)
                return torch.cat((
                    torch.zeros((1, 1, 3), device=tensor.device), patches
                ), dim=1)

        class Head(torch.nn.Module):
            def forward(self, tokens):
                logits = torch.tensor(
                    [[[2., 0.], [0., 2.], [2., 0.], [0., 2.]]],
                    device=tokens.device,
                )
                return tokens[:, 0], logits

        image = Image.new("RGB", (20, 20), "white")
        backbone = Backbone()
        with (
            mock.patch.object(experiment, "DEVICE", "cpu"),
            mock.patch.object(experiment, "INPUT_SIZE", 4),
            mock.patch.object(experiment, "TEMPERATURE", 1.0),
        ):
            distribution = experiment.encode_region(
                image, (0, 0, 20, 20), backbone, Head(), 2
            )
        self.assertEqual(backbone.batch_shape[0], 1)
        np.testing.assert_allclose(distribution, [.5, .5], atol=1e-7)

    def test_region_evaluation_uses_equal_child_average_and_separate_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (90, 90), "white").save(image_path)
            region = experiment.CocoRegion(
                1, 10, 1, "object", str(image_path), (0, 0, 90, 90)
            )
            calls = []

            def fake_encode(image, box, backbone, head, patch_size):
                calls.append(box)
                if box == region.box:
                    return np.array([.5, .5])
                left, _, _, _ = box
                return np.array([.8, .2]) if left < 45 else np.array([.2, .8])

            with mock.patch.object(experiment, "encode_region", side_effect=fake_encode):
                rows = experiment.evaluate_region(region, "model", None, None, 16)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(calls), 1 + sum(experiment.PART_COUNTS))
            self.assertAlmostEqual(rows[0].composition_js, 0)
            self.assertAlmostEqual(rows[1].composition_js, 0)
            self.assertTrue(all(np.isfinite(row.composition_js) for row in rows))


if __name__ == "__main__":
    unittest.main()
