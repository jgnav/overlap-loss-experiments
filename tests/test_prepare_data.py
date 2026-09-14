import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from prepare_data import (
    KAGGLE_IMAGENET, IMAGENET_ARCHIVES, extract_tar, extract_zip, main,
    prepare_kaggle_imagenet, required_archives,
)


class KagglePreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='kaggle-preparation-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / KAGGLE_IMAGENET
        self.output = self.root / 'prepared'
        self.csv = ('ImageId,PredictionString\n'
                    'ILSVRC2012_val_00000001,n00000002 0 0 10 10 n00000002 1 1 5 5\n'
                    'ILSVRC2012_val_00000002,n00000001 0 0 10 10\n')
        self.images = {
            'ILSVRC/Data/CLS-LOC/train/n00000001/n00000001_1.JPEG': b'train1',
            'ILSVRC/Data/CLS-LOC/train/n00000002/n00000002_1.JPEG': b'train2',
            'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG': b'val1',
            'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000002.JPEG': b'val2',
            'ILSVRC/Data/CLS-LOC/test/unused.JPEG': b'test',
        }

    def write_archive(self, nested=False):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            archive.writestr('LOC_val_solution.csv', self.csv)
            if nested:
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode='w:gz') as tar:
                    for name, contents in self.images.items():
                        entry = tarfile.TarInfo(name)
                        entry.size = len(contents)
                        tar.addfile(entry, io.BytesIO(contents))
                archive.writestr('imagenet_object_localization_patched2019.tar.gz', data.getvalue())
            else:
                for name, contents in self.images.items():
                    archive.writestr(name, contents)

    def check_layout(self):
        self.assertEqual((self.output / 'val/n00000002/ILSVRC2012_val_00000001.JPEG').read_bytes(), b'val1')
        self.assertEqual((self.output / 'val/n00000001/ILSVRC2012_val_00000002.JPEG').read_bytes(), b'val2')
        self.assertEqual(len(list(self.output.rglob('*.JPEG'))), 4)
        self.assertFalse((self.output / 'test').exists())
        self.assertTrue(self.archive.exists())

    def test_direct_zip(self):
        self.write_archive()
        prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))
        self.check_layout()

    def test_nested_tar(self):
        self.write_archive(nested=True)
        prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))
        self.check_layout()

    def test_missing_validation_image(self):
        del self.images['ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000002.JPEG']
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))

    def test_ambiguous_label(self):
        self.csv = self.csv.replace('n00000002 1 1', 'n00000001 1 1')
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))
        self.assertFalse(self.output.exists())

    def test_duplicate_label(self):
        self.csv += 'ILSVRC2012_val_00000001,n00000002 0 0 10 10\n'
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))

    def test_existing_destination_is_resumed(self):
        self.write_archive()
        self.output.mkdir()
        prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))
        self.check_layout()

    def test_interrupted_destination_is_resumed(self):
        self.write_archive()
        prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))
        (self.output / 'train/n00000002/n00000002_1.JPEG').unlink()
        prepare_kaggle_imagenet(self.archive, self.output, expected=(2, 2, 2))
        self.check_layout()

    def test_traversal(self):
        self.images['../ILSVRC/Data/CLS-LOC/train/n00000001/escape.JPEG'] = b'bad'
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            prepare_kaggle_imagenet(self.archive, self.output, expected=(3, 2, 2))

    def test_original_format_remains_supported(self):
        self.assertTrue(set(IMAGENET_ARCHIVES) <= set(required_archives(self.root)))
        self.write_archive()
        self.assertIn(KAGGLE_IMAGENET, required_archives(self.root))
        self.assertFalse(set(IMAGENET_ARCHIVES) & set(required_archives(self.root)))

    def test_cli_prepares_in_input_folder(self):
        with patch('sys.argv', ['prepare_data.py', str(self.root)]), patch('prepare_data.prepare') as prepare:
            main()
        prepare.assert_called_once_with(self.root.resolve(), self.root.resolve())


class ArchiveExtractionResumeTest(unittest.TestCase):
    def test_tar_and_zip_extraction_are_idempotent(self):
        with tempfile.TemporaryDirectory(prefix='archive-resume-') as directory:
            root = Path(directory)
            tar_archive = root / 'sample.tar'
            with tarfile.open(tar_archive, 'w') as archive:
                contents = b'tar payload'
                member = tarfile.TarInfo('nested/tar.txt')
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
            extract_tar(tar_archive, root / 'output')
            extract_tar(tar_archive, root / 'output')

            zip_archive = root / 'sample.zip'
            with zipfile.ZipFile(zip_archive, 'w') as archive:
                archive.writestr('nested/zip.txt', b'zip payload')
            extract_zip(zip_archive, root / 'output')
            extract_zip(zip_archive, root / 'output')

            self.assertEqual((root / 'output/nested/tar.txt').read_bytes(), b'tar payload')
            self.assertEqual((root / 'output/nested/zip.txt').read_bytes(), b'zip payload')


if __name__ == '__main__':
    unittest.main()
