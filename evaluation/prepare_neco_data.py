"""Prepare official VOC/SBD and COCO labels for the pinned NeCo loaders.

Images are symlinked, never duplicated. Completion is atomic and audited.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


def download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    subprocess.run(['wget', '-c', '--tries=5', '--timeout=60', '-O', str(part), url], check=True)
    part.replace(dest)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def prepare_voc(data, root, vendor):
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat
    voc = data / 'pascal_voc/VOCdevkit/VOC2012'
    target = root / 'voc'
    target.mkdir(parents=True, exist_ok=True)
    # NeCo appends VOCSegmentation; Hummingbird takes the dataset root directly.
    link(target, root / 'VOCSegmentation')
    link(voc / 'JPEGImages', target / 'images')
    link(voc / 'SegmentationClass', target / 'SegmentationClass')
    (target / 'sets').mkdir(exist_ok=True)
    train = vendor / 'hummingbird/file_sets/voc/full/trainaug.txt'
    shutil.copyfile(train, target / 'sets/trainaug.txt')
    for split in ('train', 'val'):
        shutil.copyfile(voc / f'ImageSets/Segmentation/{split}.txt', target / f'sets/{split}.txt')
    train_ids = train.read_text().split()
    val_ids = (target / 'sets/val.txt').read_text().split()
    assert len(train_ids) == 10582 and len(val_ids) == 1449
    assert not set(train_ids) & set(val_ids)
    aug = target / 'SegmentationClassAug'
    aug.mkdir(exist_ok=True)
    missing = [i for i in train_ids if not (aug / f'{i}.png').exists()]
    if missing:
        archive = root / 'downloads/benchmark.tgz'
        download('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz', archive)
        needed = {f'benchmark_RELEASE/dataset/cls/{i}.mat' for i in missing
                  if not (voc / f'SegmentationClass/{i}.png').exists()}
        with tarfile.open(archive) as tf:
            for member in tf:
                if member.name in needed:
                    mat = loadmat(tf.extractfile(member), struct_as_record=False, squeeze_me=True)
                    mask = mat['GTcls'].Segmentation.astype(np.uint8)
                    assert set(np.unique(mask)) <= set(range(21)) | {255}
                    Image.fromarray(mask).save(aug / (Path(member.name).stem + '.png'))
        for i in missing:
            original = voc / f'SegmentationClass/{i}.png'
            if original.exists():
                shutil.copyfile(original, aug / f'{i}.png')
    for i in train_ids:
        assert (target / f'images/{i}.jpg').is_file()
        with Image.open(aug / f'{i}.png') as m, Image.open(target / f'images/{i}.jpg') as im:
            assert m.size == im.size
    return {'trainaug': len(train_ids), 'validation': len(val_ids),
            'train_list_sha256': hashlib.sha256(train.read_bytes()).hexdigest()}


def prepare_coco(data, root):
    import numpy as np
    from PIL import Image
    target = root / 'coco'
    target.mkdir(parents=True, exist_ok=True)
    link(data / 'coco/images', target / 'images')
    download_root = root / 'downloads'
    pan = download_root / 'panoptic_annotations_trainval2017.zip'
    stuff = download_root / 'stuff_trainval2017.zip'
    pixel = download_root / 'stuffthingmaps_trainval2017.zip'
    download('https://s3.amazonaws.com/images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip', pan)
    download('http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip', stuff)
    download('http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip', pixel)
    raw = root / 'raw_coco'
    raw.mkdir(exist_ok=True)
    ann = target / 'annotations'
    (ann / 'panoptic_annotations').mkdir(parents=True, exist_ok=True)
    (ann / 'stuff_annotations').mkdir(exist_ok=True)
    with zipfile.ZipFile(pan) as z:
        for split in ('train', 'val'):
            name = f'panoptic_{split}2017.json'
            member = next(n for n in z.namelist() if n.endswith('/' + name) or n == name)
            (ann / 'panoptic_annotations' / name).write_bytes(z.read(member))
            nested = next(n for n in z.namelist() if n.endswith(f'panoptic_{split}2017.zip'))
            nested_path = raw / f'panoptic_{split}2017.zip'
            if not nested_path.exists():
                with z.open(nested) as inp, nested_path.open('wb') as out:
                    shutil.copyfileobj(inp, out)
    with zipfile.ZipFile(stuff) as z:
        for split in ('train', 'val'):
            name = f'stuff_{split}2017.json'
            member = next(n for n in z.namelist() if n.endswith('/' + name) or n == name)
            with z.open(member) as inp, (ann / 'stuff_annotations' / name).open('wb') as out:
                shutil.copyfileobj(inp, out)
    with zipfile.ZipFile(pixel) as z:
        for split, count in [('train', 118287), ('val', 5000)]:
            dst = ann / f'stuff_annotations/stuff_{split}2017_pixelmaps'
            dst.mkdir(exist_ok=True)
            members = [n for n in z.namelist() if f'{split}2017/' in n and n.endswith('.png')]
            assert len(members) == count, (split, len(members))
            for j, member in enumerate(members):
                out = dst / Path(member).name
                if not out.exists():
                    with z.open(member) as f:
                        labels = np.asarray(Image.open(f), dtype=np.uint16) + 1
                    # Official stuffthingmaps use category_id-1, with void=255.
                    labels[(labels < 92) | (labels > 182)] = 183
                    Image.fromarray(labels.astype(np.uint8)).save(out)
                if j % 10000 == 0:
                    print(f'Stuff {split}: {j}/{count}', flush=True)
    for split, count in [('train', 118287), ('val', 5000)]:
        meta = json.loads((ann / f'panoptic_annotations/panoptic_{split}2017.json').read_text())
        categories = meta['categories']
        assert len({c['supercategory'] for c in categories if c['isthing']}) == 12
        dst = ann / f'{split}2017'
        dst.mkdir(exist_ok=True)
        with zipfile.ZipFile(raw / f'panoptic_{split}2017.zip') as z:
            names = {Path(n).name: n for n in z.namelist() if n.endswith('.png')}
            assert len(meta['annotations']) == count
            for j, item in enumerate(meta['annotations']):
                out = dst / item['file_name']
                if not out.exists():
                    with z.open(names[item['file_name']]) as f:
                        rgb = np.asarray(Image.open(f), dtype=np.uint32)
                    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
                    # The official panoptic-to-semantic conversion: category IDs,
                    # void=255. Preserve both things and stuff; NeCo filters stuff.
                    lut = np.full(int(ids.max()) + 1, 255, dtype=np.uint8)
                    for seg in item['segments_info']:
                        lut[seg['id']] = seg['category_id']
                    Image.fromarray(lut[ids]).save(out)
                image = target / f'images/{split}2017' / Path(item['file_name']).with_suffix('.jpg')
                assert image.is_file(), image
                assert (ann / f'stuff_annotations/stuff_{split}2017_pixelmaps' / item['file_name']).is_file()
                if j % 10000 == 0:
                    print(f'Things {split}: {j}/{count}', flush=True)
        stuff_meta = json.loads((ann / f'stuff_annotations/stuff_{split}2017.json').read_text())
        assert len({c['supercategory'] for c in stuff_meta['categories']} - {'other'}) == 15
    return {'train': 118287, 'validation': 5000, 'thing_classes': 12, 'stuff_classes': 15,
            'labels': 'official COCO2017 panoptic category masks and COCO-Stuff2017 pixelmaps'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', choices=['voc', 'coco'], required=True)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    fn = prepare_voc if a.task == 'voc' else prepare_coco
    extra = [Path(__file__).resolve().parent / 'vendor'] if a.task == 'voc' else []
    report = fn(a.datasets, a.root, *extra)
    tmp = a.root / f'{a.task}_ready.json.tmp'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(a.root / f'{a.task}_ready.json')
