"""Prepare downloaded datasets and resume safely after an interrupted run."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
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
STATE_FILENAME = '.prepare_data_state.json'


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
    if output.exists() and not output.is_dir():
        raise FileExistsError(f'ImageNet output is not a directory: {output}')
    with zipfile.ZipFile(archive) as source:
        labels = kaggle_labels(source)
        if len(labels) != expected[1]:
            raise ValueError(f'Expected {expected[1]} validation labels, got {len(labels)}')
        output.mkdir(parents=True, exist_ok=True)
        seen, classes, counts, validation_ids = _scan_kaggle_output(output, labels)
        archive_seen = set()
        if sum(counts):
            print(
                f'Resuming ImageNet: found {counts[0]} train / {counts[1]} val images already present',
                flush=True)

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
            if target in archive_seen:
                raise ValueError(f'Duplicate ImageNet image: {target}')
            archive_seen.add(target)
            already_present = target in seen
            if not already_present:
                seen.add(target)
                counts[split] += 1
            return safe_destination(output, target), already_present

        def copy_image(raw, target, already_present, expected_size=None):
            if (already_present and target.is_file()
                    and (expected_size is None or target.stat().st_size == expected_size)):
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_stream(raw, target)
            if sum(counts) % 10000 == 0:
                print(f'ImageNet: {counts[0]} train / {counts[1]} val extracted', flush=True)

        members = source.infolist()
        direct = any('ILSVRC/Data/CLS-LOC/train/' in m.filename and m.filename.endswith('.JPEG') for m in members)
        if direct:
            for member in members:
                target_info = image_path(member.filename)
                if target_info is None:
                    continue
                target, already_present = target_info
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f'Image symlink rejected: {member.filename}')
                with source.open(member) as raw:
                    copy_image(raw, target, already_present, member.file_size)
        else:
            nested = [m for m in members if m.filename.endswith(('.tar.gz', '.tgz', '.tar'))]
            if len(nested) != 1:
                raise ValueError('Expected ILSVRC/Data/CLS-LOC images or one nested ImageNet tar in Kaggle ZIP')
            # Avoid an extra ~150 GB intermediate tar on disk.
            with source.open(nested[0]) as raw, tarfile.open(fileobj=raw, mode='r|*') as tar:
                for member in tar:
                    target_info = image_path(member.name)
                    if target_info is None:
                        continue
                    target, already_present = target_info
                    if not member.isfile():
                        raise ValueError(f'Image is not a regular file: {member.name}')
                    if (already_present and target.is_file()
                            and target.stat().st_size == member.size):
                        continue
                    with tar.extractfile(member) as image:
                        copy_image(image, target, already_present, member.size)
        if (tuple(counts) != expected[:2] or len(classes) != expected[2]
                or validation_ids != set(labels) or set(labels.values()) != classes):
            raise ValueError(f'Incomplete/inconsistent ImageNet: train={counts[0]}, val={counts[1]}, classes={len(classes)}')
    print('Kaggle ImageNet train/val prepared; test images skipped.', flush=True)


def _scan_kaggle_output(output, labels):
    """Recover counts and targets already written by an interrupted Kaggle run."""
    seen, classes, counts, validation_ids = set(), set(), [0, 0], set()
    if not output.exists():
        return seen, classes, counts, validation_ids
    for path in output.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'ImageNet output symlink rejected: {path}')
        if not path.is_file() or path.suffix != '.JPEG':
            continue
        relative = path.relative_to(output).as_posix()
        train_match = re.fullmatch(r'train/(n\d{8})/([^/]+\.JPEG)', relative)
        val_match = re.fullmatch(r'val/(n\d{8})/(ILSVRC2012_val_\d{8})\.JPEG', relative)
        if train_match:
            target = relative
            classes.add(train_match[1])
            split = 0
        elif val_match:
            synset, image_id = val_match.groups()
            if image_id not in labels or labels[image_id] != synset:
                raise ValueError(f'Unexpected validation image in existing output: {relative}')
            target = relative
            validation_ids.add(image_id)
            split = 1
        else:
            raise ValueError(f'Unexpected ImageNet image in existing output: {relative}')
        if target in seen:
            raise ValueError(f'Duplicate ImageNet image: {target}')
        seen.add(target)
        counts[split] += 1
    for image_id in validation_ids:
        classes.add(labels[image_id])
    return seen, classes, counts, validation_ids


def _copy_stream(stream, target):
    """Atomically copy a stream, leaving no partially written destination file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f'Refusing to overwrite symlink: {target}')
    if target.exists() and not target.is_file():
        raise ValueError(f'Refusing to overwrite non-file: {target}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='wb', dir=target.parent, prefix=f'.{target.name}.', suffix='.part',
                delete=False) as handle:
            temporary = Path(handle.name)
            shutil.copyfileobj(stream, handle, length=1024 * 1024)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def safe_destination(root, name):
    root = Path(root).expanduser().resolve()
    relative = Path(name)
    destination = root / relative
    resolved = destination.resolve()
    if relative.is_absolute() or (resolved != root and root not in resolved.parents):
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
        for member in members:
            target = safe_destination(destination, member.filename)
            if member.is_dir() or member.filename.endswith('/'):
                if target.is_symlink():
                    raise ValueError(f'Archive symlink is not supported: {member.filename}')
                target.mkdir(parents=True, exist_ok=True)
                continue
            with source.open(member) as stream:
                if (target.is_file() and target.stat().st_size == member.file_size
                        and not target.is_symlink()):
                    continue
                _copy_stream(stream, target)


def extract_tar(archive, destination):
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            safe_destination(destination, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsupported archive entry: {member.name}')
        for member in members:
            target = safe_destination(destination, member.name)
            if member.isdir():
                if target.is_symlink():
                    raise ValueError(f'Archive symlink is not supported: {member.name}')
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (target.is_file() and target.stat().st_size == member.size
                    and not target.is_symlink()):
                continue
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f'Unable to read archive member: {member.name}')
            with stream:
                _copy_stream(stream, target)


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


def _archive_fingerprint(archive_dir, archives):
    return {
        name: {
            'size': (archive_dir / name).stat().st_size,
            'mtime_ns': (archive_dir / name).stat().st_mtime_ns,
        }
        for name in archives
    }


def _load_state(path, fingerprint):
    if not path.is_file():
        return {'version': 1, 'archives': fingerprint, 'completed': {}}
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        print(f'Ignoring unreadable preparation state: {path}', flush=True)
        return {'version': 1, 'archives': fingerprint, 'completed': {}}
    if (not isinstance(state, dict)
            or state.get('version') != 1
            or state.get('archives') != fingerprint
            or not isinstance(state.get('completed'), dict)):
        print('Archive set changed; rechecking all preparation phases.', flush=True)
        return {'version': 1, 'archives': fingerprint, 'completed': {}}
    return state


def _save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _run_phase(name, state, state_path, output, paths, action):
    completed = state.setdefault('completed', {})
    phase_paths = [output / path for path in paths]
    if completed.get(name) and all(path.exists() for path in phase_paths):
        print(f'Skipping completed phase: {name}', flush=True)
        return
    print(f'Running phase: {name}', flush=True)
    action()
    completed[name] = True
    _save_state(state_path, state)


def prepare(archive_dir, output):
    archives = required_archives(archive_dir)
    kaggle = KAGGLE_IMAGENET in archives
    missing = [name for name in archives if not (archive_dir / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing archives in {}: {}'.format(archive_dir, ', '.join(missing)))

    from torchvision.datasets import ImageNet
    from torchvision.datasets.imagenet import (
        ARCHIVE_META, parse_devkit_archive, parse_train_archive, parse_val_archive,
    )
    from torchvision.datasets.utils import check_integrity
    from evaluation.prepare_voc_manifest import prepare_voc2012_manifest
    from evaluation.prepare_coco_manifest import prepare_coco2017_manifest

    output.mkdir(parents=True, exist_ok=True)
    state_path = output / STATE_FILENAME
    state = _load_state(state_path, _archive_fingerprint(archive_dir, archives))
    existing = [name for name in DATA_DIRECTORIES
                if (output / name).exists() or (output / name).is_symlink()]
    if existing:
        print(f'Resuming dataset preparation in {output}', flush=True)
        print(f'Existing dataset components: {", ".join(existing)}', flush=True)
    else:
        print(f'Starting dataset preparation in {output}', flush=True)
    completed = [name for name, done in state.get('completed', {}).items() if done]
    if completed:
        print(f'Completed phases from state: {", ".join(completed)}', flush=True)
    _save_state(state_path, state)

    def verify_archives():
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

    _run_phase('integrity', state, state_path, output, (), verify_archives)

    _run_phase(
        'pascal_voc', state, state_path, output, ('pascal_voc',),
        lambda: extract_tar(archive_dir / 'VOCtrainval_11-May-2012.tar', output / 'pascal_voc'))
    _run_phase(
        'ade20k', state, state_path, output, ('ade20k',),
        lambda: extract_zip(archive_dir / 'ADEChallengeData2016.zip', output / 'ade20k'))
    _run_phase(
        'coco_images', state, state_path, output, ('coco/images',),
        lambda: [extract_zip(archive_dir / f'{split}2017.zip', output / 'coco/images')
                 for split in ('train', 'val')])
    _run_phase(
        'coco_annotations', state, state_path, output, ('coco/annotations',),
        lambda: extract_zip(
            archive_dir / 'annotations_trainval2017.zip', output / 'coco',
            ('annotations/instances_train2017.json', 'annotations/instances_val2017.json')))
    _run_phase(
        'cityscapes_images', state, state_path, output, ('cityscapes/leftImg8bit',),
        lambda: extract_zip(
            archive_dir / 'leftImg8bit_trainvaltest.zip', output / 'cityscapes',
            ('leftImg8bit/train/', 'leftImg8bit/val/')))
    _run_phase(
        'cityscapes_masks', state, state_path, output, ('cityscapes/gtFine',),
        lambda: extract_zip(
            archive_dir / 'gtFine_trainvaltest.zip', output / 'cityscapes',
            ('gtFine/train/', 'gtFine/val/')))

    print('Preparing ImageNet classes (this can take hours)...', flush=True)
    imagenet = output / 'imagenet'
    # Absolute archive filenames let torchvision read from any source folder.
    # No copying or symlinking of the large downloaded archives is needed.
    def prepare_imagenet():
        if kaggle:
            prepare_kaggle_imagenet(archive_dir / KAGGLE_IMAGENET, imagenet)
        else:
            imagenet.mkdir(parents=True, exist_ok=True)
            parse_devkit_archive(str(imagenet), file=str(archive_dir / ARCHIVE_META['devkit'][0]))
            parse_train_archive(str(imagenet), file=str(archive_dir / ARCHIVE_META['train'][0]))
            parse_val_archive(str(imagenet), file=str(archive_dir / ARCHIVE_META['val'][0]))
            for split in ('train', 'val'):
                ImageNet(str(imagenet), split=split)

    _run_phase('imagenet', state, state_path, output, ('imagenet',), prepare_imagenet)

    print('Generating manifests and validating data...', flush=True)
    _run_phase(
        'manifests', state, state_path, output, ('evaluation_manifests',),
        lambda: (prepare_voc2012_manifest(output), prepare_coco2017_manifest(output)))
    validate(output)
    state['validated'] = True
    _save_state(state_path, state)
    print(f'Data ready: {output}\nOriginal archives retained.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_path', type=Path, help='Archive folder (Kaggle ImageNet ZIP or original ImageNet tars)')
    args = parser.parse_args()
    archive_dir = args.dataset_path.expanduser().resolve()
    prepare(archive_dir, archive_dir)


if __name__ == '__main__':
    main()
