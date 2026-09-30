"""Prepare the local YouTube-VOS 2019 and MOSEv2 releases without downloading.

Run on a compute node using slurm/prepare_downloaded_vos.sh. Public validation
initialization masks and sample predictions are never treated as scoring masks.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zipfile


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def run(command):
    log(' '.join(map(str, command)))
    subprocess.run(list(map(str, command)), check=True)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def unpack(archive, destination, marker):
    """Stage extraction so a failed archive never publishes a partial split."""
    if marker.is_file():
        log(f'Already extracted: {archive.name}')
        return
    stage = destination.parent / (destination.name + '.preparing')
    stage.mkdir(parents=True, exist_ok=True)
    if archive.suffix == '.zip':
        run(['unzip', '-oq', archive, '-d', stage])
    elif archive.suffix == '.7z':
        run(['bsdtar', '-xf', archive, '-C', stage, '--no-same-owner'])
    else:
        run(['tar', '-xf', archive, '-C', stage, '--no-same-owner'])
    destination.mkdir(parents=True, exist_ok=True)
    # Move whole top-level directories where possible; merge only overlapping
    # folders (e.g. all-frame RGB and the labeled-frame RGB release).
    merge(stage, destination)
    stage.rmdir()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(archive) + '\n')


def merge(source, destination):
    for path in source.iterdir():
        target = destination / path.name
        if not target.exists():
            path.rename(target)
        elif path.is_dir() and target.is_dir():
            merge(path, target)
            path.rmdir()
        elif path.is_file() and target.is_file():
            # Archive overlap is intentional. Preserve the latest extracted
            # release file, without touching files outside its destination.
            path.replace(target)
        else:
            raise ValueError(f'Conflicting extraction paths: {path}, {target}')


def child_archive(raw, work, outer, member):
    target = work / Path(member).name
    marker = work / (target.name + '.zip_crc_ok')
    if target.is_file() and marker.is_file():
        return target
    with zipfile.ZipFile(raw / outer) as archive:
        info = archive.getinfo(member)
        partial = target.with_name(target.name + '.partial')
        with archive.open(info) as source, partial.open('wb') as output:
            shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        if partial.stat().st_size != info.file_size:
            raise ValueError(f'Incomplete nested archive: {member}')
        partial.replace(target)
        marker.write_text(f'{outer}:{member}\n')
    log(f'CRC verified nested archive: {member}')
    return target


def joined_archive(first, second, work, filename):
    with first.open('rb') as source:
        header = source.read(32)
    if header[:6] != b'7z\xbc\xaf\x27\x1c':
        raise ValueError(f'Not a 7z first volume: {first}')
    _, offset, size, _ = struct.unpack('<IQQI', header[8:])
    expected = 32 + offset + size
    if first.stat().st_size + second.stat().st_size != expected:
        raise ValueError(f'Missing/truncated archive volume: {first}')
    target = work / filename
    if target.is_file() and target.stat().st_size == expected:
        return target
    partial = target.with_name(target.name + '.partial')
    with partial.open('wb') as output:
        for part in (first, second):
            with part.open('rb') as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    partial.replace(target)
    log(f'Joined complete volumes: {filename} ({expected} bytes)')
    return target


def inventory(base, metadata=None, youtube=False):
    images, annotations = base / 'JPEGImages', base / 'Annotations'
    names = sorted(p.name for p in images.iterdir() if p.is_dir())
    expected = metadata.get('videos', {}) if metadata else {}
    if expected and set(names) != set(expected):
        raise ValueError(f'Video names disagree with metadata: {base}')
    report = dict(videos=len(names), images=0, masks=0, missing_rgb=0,
                  missing_initial_masks=0, missing_scoring_masks=0)
    for name in names:
        frames = sorted((images / name).glob('*.jpg'))
        masks = {p.stem for p in (annotations / name).glob('*.png')}
        stems = {p.stem for p in frames}
        report['images'] += len(frames)
        report['masks'] += len(masks)
        if not frames:
            raise ValueError(f'Empty video: {images / name}')
        report['missing_initial_masks'] += frames[0].stem not in masks
        video = expected.get(name, {})
        if youtube:
            required = {f for obj in video.get('objects', {}).values()
                        for f in obj.get('frames', [])}
            report['missing_rgb'] += len(required - stems)
            report['missing_scoring_masks'] += len(required - masks)
        else:
            required = {Path(f).stem for f in video.get('frames', [])}
            report['missing_rgb'] += len(required - stems)
            report['missing_scoring_masks'] += len(stems - masks)
    if report['missing_rgb'] or report['missing_initial_masks']:
        raise ValueError(f'Incomplete released RGB/initialization data: {base}: {report}')
    report['offline_scoring_ready'] = report['missing_scoring_masks'] == 0
    splits = base / 'ImageSets'
    splits.mkdir(exist_ok=True)
    (splits / 'val.txt').write_text('\n'.join(names) + '\n')
    return report


def prepare_youtube(root, raw, work):
    destination = root / 'youtube_vos_2019'
    archives = [raw / 'train-007.tar']
    archives.append(child_archive(raw, work, 'drive-download-20260929T173327Z-1-006.zip', 'valid.tar'))
    archives.append(child_archive(raw, work, 'drive-download-20260929T173327Z-1-001.zip', 'test.zip'))
    for archive in archives:
        unpack(archive, destination, work / (archive.name + '.extracted'))
    for split, suffix, outer in [('valid', '003', '004'), ('test', '002', '005')]:
        member = f'{split}_all_frames_zip/{split}_all_frames.7z.002'
        second = child_archive(raw, work, f'drive-download-20260929T173327Z-1-{outer}.zip', member)
        archive = joined_archive(raw / f'{split}_all_frames.7z-{suffix}.001', second, work,
                                 f'{split}_all_frames.7z')
        # Keep all-frame releases separate: their frame set differs from the
        # official labeled-frame split used by the current evaluator.
        unpack(archive, destination / 'all_frames' / split,
               work / (archive.name + '.extracted'))
    gt = child_archive(raw, work, 'drive-download-20260929T173327Z-1-004.zip', 'test_gt.zip')
    unpack(gt, destination / 'released_test_gt', work / 'test_gt.extracted')
    scoring = child_archive(raw, work, 'drive-download-20260929T173327Z-1-001.zip',
                            'scoring_program_release.zip')
    unpack(scoring, destination / 'scoring_program', work / 'scoring_program.extracted')
    metadata = json.loads((destination / 'valid/meta.json').read_text())
    report = inventory(destination / 'valid', metadata, youtube=True)
    report['version'] = 'YouTube-VOS 2019'
    report['path'] = str(destination)
    log(f'YouTube-VOS validation inventory: {report}')
    return report


def prepare_mose(root, raw, work):
    destination = root / 'MOSEv2'
    bundle = raw / 'drive-download-20260929T173810Z-1-003.zip'
    with zipfile.ZipFile(bundle) as source:
        checksums = dict((name, sha) for sha, name in
                         (line.split() for line in source.read('SHA256SUMS').decode().splitlines()))
        metadata = {split: source.read(f'meta_{split}.json') for split in ('train', 'valid')}
    for split, name in [('valid', 'valid-002.tar.gz'), ('train', 'train-001.tar.gz')]:
        archive = raw / name
        marker = work / (name + '.sha256_ok')
        expected = checksums[f'{split}.tar.gz']
        if not marker.is_file() or marker.read_text().strip() != expected:
            log(f'Verifying official SHA256: {name}')
            if digest(archive) != expected:
                raise ValueError(f'SHA256 mismatch; source download is incomplete/corrupt: {archive}')
            marker.write_text(expected + '\n')
        unpack(archive, destination, work / (name + '.extracted'))
        (destination / f'meta_{split}.json').write_bytes(metadata[split])
    # This is example output, not ground truth; preserve it in an auxiliary
    # directory and never install it as Annotations.
    destination.mkdir(exist_ok=True)
    shutil.copy2(bundle, destination / bundle.name)
    valid_metadata = json.loads(metadata['valid'])
    report = inventory(destination / 'valid', valid_metadata)
    report['version'] = valid_metadata['info']['version']
    report['path'] = str(destination)
    log(f'MOSEv2 validation inventory: {report}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.datasets_root
    raw, work = root / 'raw', root / 'downloads/prepared_vos'
    work.mkdir(parents=True, exist_ok=True)
    report = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(function, root, raw, work)
                   for name, function in [('youtube_vos', prepare_youtube), ('mose_v2', prepare_mose)]}
        for name, future in futures.items():
            try:
                report[name] = future.result()
            except Exception as error:
                report[name] = dict(error=f'{type(error).__name__}: {error}')
                log(f'{name} failed: {error}')
            (work / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    log(f'Preparation report: {work / "preparation_report.json"}')
    if any('error' in result for result in report.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
