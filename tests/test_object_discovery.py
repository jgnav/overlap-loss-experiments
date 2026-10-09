"""Protocol checks against official TokenCut and detection annotation conventions."""
import tempfile
import unittest
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from evaluation.object_discovery import dataset_records, open_database, summarize_rows
from evaluation.utils.object_discovery import (
    THRESHOLDS, box_iou, coco_boxes, native_image_tensor, predict_boxes, voc_boxes,
)
from evaluation.vendor.tokencut.object_discovery import ncut


class ObjectDiscoveryTest(unittest.TestCase):
    def test_patch_outputs_match_official_tokencut_without_dropping_first_patch(self):
        generator = torch.Generator().manual_seed(17)
        patches = torch.randn(1, 12, 7, generator=generator)
        actual = predict_boxes(patches, [3, 4], (43, 59))
        # A real CLS of arbitrary magnitude must not enter the patch graph.
        reference = torch.cat((torch.full_like(patches[:, :1], 10000), patches), dim=1)
        for threshold in THRESHOLDS:
            box, _, _, seed, _, _ = ncut(reference, [3, 4], [16, 16], (3, 43, 59), tau=threshold)
            np.testing.assert_array_equal(actual[f"{threshold:.2f}"]["bbox_xyxy"], box)
            self.assertEqual(actual[f"{threshold:.2f}"]["seed_patch"], seed)

    def test_original_pixels_are_preserved_and_padding_is_normalized_zero(self):
        image = Image.new("RGB", (29, 17), (0, 128, 255))
        tensor, original = native_image_tensor(image)
        self.assertEqual(original, (17, 29))
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        expected = torch.tensor([(0 - .485) / .229, (128 / 255 - .456) / .224, (1 - .406) / .225])
        torch.testing.assert_close(tensor[:, 16, 28], expected)
        self.assertEqual(torch.count_nonzero(tensor[:, 17:, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(tensor[:, :, 29:]).item(), 0)

    def test_gt_coordinates_and_crowd_handling_match_official_release(self):
        annotation = ET.fromstring('<annotation><object><difficult>1</difficult><truncated>1</truncated>'
                                   '<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax>'
                                   '</bndbox></object></annotation>')
        self.assertEqual(voc_boxes(annotation), [[0, 1, 20, 30]])
        self.assertEqual(coco_boxes([{'bbox': [1.2, 2.1, 3.2, 4.3], 'iscrowd': 0},
                                    {'bbox': [0, 0, 100, 100], 'iscrowd': 1}]), [[1, 2, 4, 6]])
        np.testing.assert_array_equal(box_iou([0, 0, 10, 10], [[0, 0, 20, 10]]), [0.5])
        self.assertEqual(len(box_iou([0, 0, 10, 10], [])), 0)

    def test_sweep_uses_one_threshold_per_dataset_and_distinguishes_iou_boundary(self):
        rows = [{f"{t:.2f}": {"max_iou": 0.0} for t in THRESHOLDS} for _ in range(2)]
        rows[0]['0.00']['max_iou'] = 0.75
        rows[1]['0.05']['max_iou'] = 0.75
        rows[0]['0.05']['max_iou'] = 0.5
        result = summarize_rows(rows, 2)
        self.assertEqual(result['best']['corloc'], 50)
        self.assertEqual(result['threshold_results'][1]['corloc_ge_0_5'], 100)
        self.assertIsNone(summarize_rows(rows[:1], 2)['best'])

    def test_resume_rejects_a_different_evaluation_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.sqlite3'
            db = open_database(path, 'first')
            with db:
                db.execute('INSERT INTO predictions VALUES (?, ?)', ('img1', '{}'))
            db.close()
            db = open_database(path, 'first')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0], 1)
            db.close()
            with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                open_database(path, 'second')

    def test_official_coco_prefix_maps_2014_ids_to_existing_2017_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for year in ('2007', '2012'):
                voc = root / 'pascal_voc/VOCdevkit' / ('VOC' + year)
                for path in ('ImageSets/Main', 'Annotations', 'JPEGImages'):
                    (voc / path).mkdir(parents=True)
                (voc / 'ImageSets/Main/trainval.txt').write_text('img1\n')
                (voc / 'Annotations/img1.xml').write_text(
                    '<annotation><size><width>16</width><height>16</height></size>'
                    '<object><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>16</xmax>'
                    '<ymax>16</ymax></bndbox></object></annotation>')
                Image.new('RGB', (16, 16)).save(voc / 'JPEGImages/img1.jpg')
            for image_id, split in ((123, 'train2017'), (124, 'val2017')):
                path = root / 'coco/images' / split
                path.mkdir(parents=True)
                Image.new('RGB', (16, 16)).save(path / f'{image_id:012d}.jpg')
            annotation_path = root / 'coco/annotations/instances_train2014.json'
            annotation_path.parent.mkdir()
            annotation_path.write_text(json.dumps({
                'images': [{'id': i, 'file_name': f'COCO_train2014_{i:012d}.jpg', 'width': 16, 'height': 16}
                           for i in (123, 124)],
                'annotations': [{'image_id': i, 'bbox': [0, 0, 16, 16], 'iscrowd': int(i == 124)}
                                for i in (123, 124)]}))
            subset = root / 'evaluation/vendor/tokencut/coco_20k_filenames.txt'
            subset.parent.mkdir(parents=True)
            subset.write_text(''.join(f'train2014/COCO_train2014_{i:012d}.jpg\n' for i in (123, 124)))
            with patch('evaluation.object_discovery.PROJECT', root), \
                    patch('evaluation.object_discovery.EXPECTED', {'VOC07': 1, 'VOC12': 1, 'COCO20K': 2}):
                records, inputs = dataset_records({'datasets_root': str(root)})
            self.assertEqual([row['id'] for row in records['COCO20K']], ['123', '124'])
            self.assertIn('/train2017/', records['COCO20K'][0]['image'])
            self.assertIn('/val2017/', records['COCO20K'][1]['image'])
            self.assertEqual(records['COCO20K'][0]['gt_boxes_xyxy'], [[0, 0, 16, 16]])
            self.assertEqual(records['COCO20K'][1]['gt_boxes_xyxy'], [])
            self.assertEqual(inputs['COCO20K']['images_without_noncrowd_boxes'], 1)


if __name__ == '__main__':
    unittest.main()
