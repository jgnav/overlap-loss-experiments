"""Prepare downloaded datasets, accepting Kaggle or original ImageNet archives."""

import argparse
import csv
import io
from pathlib import Path
import re
import shutil
import tarfile
import zipfile


IMAGENET_ARCHIVES = (
    'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar',
    'ILSVRC2012_devkit_t12.tar.gz',
)
KAGGLE_IMAGENET = 'imagenet-object-localization-challenge.zip'
ARCHIVES = (
    'ADEChallengeData2016.zip',
    'VOCtrainval_11-May-2012.tar', 'train2017.zip', 'val2017.zip',
    'annotations_trainval2017.zip', 'leftImg8bit_trainvaltest.zip',
    'gtFine_trainvaltest.zip',
)
DATA_DIRECTORIES = ('imagenet', 'ade20k', 'pascal_voc', 'cityscapes', 'coco', 'evaluation_manifests')


def required_archives(archive_dir):
    imagenet = (KAGGLE_IMAGENET,) if (archive_dir / KAGGLE_IMAGENET).is_file() else IMAGENET_ARCHIVES
    return ARCHIVES + imagenet


def kaggle_labels(source):
    matches = [m for m in source.infolist() if Path(m.filename).name == 'LOC_val_solution.csv']
    if len(matches) != 1:
        raise ValueError('Kaggle ZIP must contain exactly one LOC_val_solution.csv')
    labels = {}
    with source.open(matches[0]) as raw:
        rows = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
        if not {'ImageId', 'PredictionString'} <= set(rows.fieldnames or []):
            raise ValueError('Invalid LOC_val_solution.csv columns')
        for row in rows:
            image_id = row['ImageId']
            tokens = (row['PredictionString'] or '').split()
            classes = set(tokens[::5])
            if (not re.fullmatch(r'ILSVRC2012_val_\d{8}', image_id or '')
                    or image_id in labels or not tokens or len(tokens) % 5
                    or len(classes) != 1
                    or not re.fullmatch(r'n\d{8}', tokens[0])):
                raise ValueError(f'Invalid/duplicate/ambiguous validation label: {image_id}')
            # Several boxes of the same class are still one classification label.
            labels[image_id] = tokens[0]
    return labels


def prepare_kaggle_imagenet(archive, output, expected=(1281167, 50000, 1000)):
    """Stream images from either a flat Kaggle ZIP or its nested ImageNet tar."""
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    with zipfile.ZipFile(archive) as source:
        labels = kaggle_labels(source)
        if len(labels) != expected[1]:
            raise ValueError(f'Expected {expected[1]} validation labels, got {len(labels)}')
        seen, classes, counts, validation_ids = set(), set(), [0, 0], set()

        def image_path(name):
            marker = 'ILSVRC/Data/CLS-LOC/'
            # Optional archive wrapper directory is allowed.
            index = name.find(marker)
            if index < 0 or (index and name[index - 1] != '/'):
                return None
            relative = name[index + len(marker):]
            if not relative.startswith(('train/', 'val/')) or not relative.lower().endswith('.jpeg'):
                return None
            safe_destination(output, name)
            train_match = re.fullmatch(r'train/(n\d{8})/([^/]+\.JPEG)', relative)
            val_match = re.fullmatch(r'val/(ILSVRC2012_val_\d{8})\.JPEG', relative)
            if train_match:
                synset, filename = train_match.groups()
                classes.add(synset)
                target = f'train/{synset}/{filename}'
                split = 0
            elif val_match:
                image_id = val_match[1]
                if image_id not in labels:
                    raise ValueError(f'Missing validation label: {image_id}')
                validation_ids.add(image_id)
                target = f'val/{labels[image_id]}/{image_id}.JPEG'
                split = 1
            else:
                raise ValueError(f'Unexpected Kaggle image layout: {name}')
            if target in seen:
                raise ValueError(f'Duplicate ImageNet image: {target}')
            seen.add(target)
            counts[split] += 1
            return safe_destination(output, target)

        def copy_image(raw, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                shutil.copyfileobj(raw, handle, length=1024 * 1024)
            if sum(counts) % 10000 == 0:
                print(f'ImageNet: {counts[0]} train / {counts[1]} val extracted', flush=True)

        members = source.infolist()
        direct = any('ILSVRC/Data/CLS-LOC/train/' in m.filename and m.filename.endswith('.JPEG') for m in members)
        if direct:
            for member in members:
                target = image_path(member.filename)
                if target is None:
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f'Image symlink rejected: {member.filename}')
                with source.open(member) as raw:
                    copy_image(raw, target)
        else:
            nested = [m for m in members if m.filename.endswith(('.tar.gz', '.tgz', '.tar'))]
            if len(nested) != 1:
                raise ValueError('Expected ILSVRC/Data/CLS-LOC images or one nested ImageNet tar in Kaggle ZIP')
            # Avoid an extra ~150 GB intermediate tar on disk.
            with source.open(nested[0]) as raw, tarfile.open(fileobj=raw, mode='r|*') as tar:
                for member in tar:
                    target = image_path(member.name)
                    if target is None:
                        continue
                    if not member.isfile():
                        raise ValueError(f'Image is not a regular file: {member.name}')
                    with tar.extractfile(member) as image:
                        copy_image(image, target)
        if (tuple(counts) != expected[:2] or len(classes) != expected[2]
                or validation_ids != set(labels) or set(labels.values()) != classes):
            raise ValueError(f'Incomplete/inconsistent ImageNet: train={counts[0]}, val={counts[1]}, classes={len(classes)}')
    print('Kaggle ImageNet train/val prepared; test images skipped.', flush=True)


def safe_destination(root, name):
    root = root.resolve()
    destination = (root / name).resolve()
    if Path(name).is_absolute() or root not in destination.parents:
        raise ValueError(f'Unsafe archive path: {name}')
    return destination


def extract_zip(archive, destination, prefixes=()):
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist()
                   if not prefixes or m.filename.startswith(prefixes)]
        if not members:
            raise ValueError(f'No expected files in {archive}')
        for member in members:
            safe_destination(destination, member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Archive symlink is not supported: {member.filename}')
        # Reading during extraction also validates each selected member's CRC.
        source.extractall(destination, members)


def extract_tar(archive, destination):
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            safe_destination(destination, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsupported archive entry: {member.name}')
        source.extractall(destination, members=members)


def validate(root):
    from torchvision.datasets import ImageFolder
    from evaluation.utils.datasets import make_ade20k, make_pascal_voc, make_cityscapes
    from evaluation.utils.classification_data import read_multilabel_manifest

    train, val = (ImageFolder(root / 'imagenet' / split) for split in ('train', 'val'))
    if (len(train), len(val), len(train.classes)) != (1281167, 50000, 1000):
        raise ValueError('Unexpected ImageNet counts')
    if train.class_to_idx != val.class_to_idx:
        raise ValueError('ImageNet train/val classes differ')
    train[0], val[0]
    print('ImageNet splits OK', flush=True)
    for factory, splits in (
        (make_pascal_voc, [('train', 1464), ('val', 1449)]),
        (make_ade20k, [('training', 20210), ('validation', 2000)]),
        (make_cityscapes, [('train', 2975), ('val', 500)]),
    ):
        for split, expected in splits:
            dataset = factory(root, split)
            if len(dataset) != expected:
                raise ValueError(f'{factory.__name__}/{split}: expected {expected}, got {len(dataset)}')
            for image, mask in zip(dataset.images, dataset.targets):
                if not image.is_file() or not mask.is_file():
                    raise FileNotFoundError(f'Missing image/mask pair: {image}, {mask}')
            dataset[0]
            print(factory.__name__, split, len(dataset), 'OK', flush=True)
    for name, classes, expected in (
        ('pascal_voc', 20, {'train': 5717, 'val': 5823}),
        ('coco', 80, {'train': 118287, 'val': 5000}),
    ):
        _, _, metadata = read_multilabel_manifest(
            root / 'evaluation_manifests' / f'{name}.json', root, name, classes)
        if metadata['split_sizes'] != expected:
            raise ValueError(f'Unexpected {name} classification split sizes')
        print(name, 'manifest OK', flush=True)


def prepare(archive_dir, output):
    archives = required_archives(archive_dir)
    kaggle = KAGGLE_IMAGENET in archives
    missing = [name for name in archives if not (archive_dir / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing archives in {}: {}'.format(archive_dir, ', '.join(missing)))
    # Never overwrite prepared data or silently reuse a partial extraction.
    existing = [name for name in DATA_DIRECTORIES
                if (output / name).exists() or (output / name).is_symlink()]
    if existing:
        raise FileExistsError(f'Use a fresh output directory; already present: {existing}')

    from torchvision.datasets import ImageNet
    from torchvision.datasets.imagenet import (
        ARCHIVE_META, parse_devkit_archive, parse_train_archive, parse_val_archive,
    )
    from torchvision.datasets.utils import check_integrity
    from evaluation.prepare_voc_manifest import prepare_voc2012_manifest
    from evaluation.prepare_coco_manifest import prepare_coco2017_manifest

    print('Checking archive integrity...', flush=True)
    checksums = ([] if kaggle else list(ARCHIVE_META.values())) + [
        ('VOCtrainval_11-May-2012.tar', '6cd6e144f989b92b3379bac3b3de84fd')]
    for name, checksum in checksums:
        if not check_integrity(str(archive_dir / name), checksum):
            raise ValueError(f'Archive checksum failed: {name}')
    for name in archives:
        if name.endswith('.zip'):
            with zipfile.ZipFile(archive_dir / name) as source:
                bad = source.testzip()
                if bad:
                    raise ValueError(f'Corrupt ZIP member: {name}: {bad}')

    output.mkdir(parents=True, exist_ok=True)
    print('Extracting VOC, ADE20K, COCO and Cityscapes...', flush=True)
    extract_tar(archive_dir / 'VOCtrainval_11-May-2012.tar', output / 'pascal_voc')
    extract_zip(archive_dir / 'ADEChallengeData2016.zip', output / 'ade20k')
    for split in ('train', 'val'):
        extract_zip(archive_dir / f'{split}2017.zip', output / 'coco/images')
    extract_zip(archive_dir / 'annotations_trainval2017.zip', output / 'coco',
                ('annotations/instances_train2017.json', 'annotations/instances_val2017.json'))
    extract_zip(archive_dir / 'leftImg8bit_trainvaltest.zip', output / 'cityscapes',
                ('leftImg8bit/train/', 'leftImg8bit/val/'))
    extract_zip(archive_dir / 'gtFine_trainvaltest.zip', output / 'cityscapes',
                ('gtFine/train/', 'gtFine/val/'))

    print('Preparing ImageNet classes (this can take hours)...', flush=True)
    imagenet = output / 'imagenet'
    # Absolute archive filenames let torchvision read from any source folder.
    # No copying or symlinking of the large downloaded archives is needed.
    if kaggle:
        prepare_kaggle_imagenet(archive_dir / KAGGLE_IMAGENET, imagenet)
    else:
        imagenet.mkdir()
        parse_devkit_archive(str(imagenet), file=str(archive_dir / ARCHIVE_META['devkit'][0]))
        parse_train_archive(str(imagenet), file=str(archive_dir / ARCHIVE_META['train'][0]))
        parse_val_archive(str(imagenet), file=str(archive_dir / ARCHIVE_META['val'][0]))
        for split in ('train', 'val'):
            ImageNet(str(imagenet), split=split)

    print('Generating manifests and validating data...', flush=True)
    prepare_voc2012_manifest(output)
    prepare_coco2017_manifest(output)
    validate(output)
    print(f'Data ready: {output}\nOriginal archives retained.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_path', type=Path, help='Archive folder (Kaggle ImageNet ZIP or original ImageNet tars)')
    args = parser.parse_args()
    archive_dir = args.dataset_path.expanduser().resolve()
    prepare(archive_dir, archive_dir)


if __name__ == '__main__':
    main()
