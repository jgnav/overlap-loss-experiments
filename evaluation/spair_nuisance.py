"""Official SPair nuisance subsets with the supplied CRISP matching function.

The table uses HPF/SPair's mean of per-pair PCK, at alpha_bbox=0.1.
The full test layout is used, without sampling or test-time augmentation.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path('/mnt/fast/nobackup/users/jg02228/overlap-loss-experiments')
CRISP = Path('/mnt/fast/nobackup/scratch4weeks/jg02228/probe3d')
MODELS = {
    'ibot_original': PROJECT / 'checkpoints/ibot_vit_small.pth',
    'region200': PROJECT / 'output/long_ibot_vit_small/85535_0/checkpoint_source1000_continuation0200.pth',
}
FACTORS = {
    'viewpoint_variation': ['Easy', 'Med.', 'Hard'],
    'scale_variation': ['Easy', 'Med.', 'Hard'],
    'truncation': ['None', 'Src.', 'Tgt.', 'Both'],
    'occlusion': ['None', 'Src.', 'Tgt.', 'Both'],
}
EXPECTED = {
    'viewpoint_variation': [6654, 4474, 1106],
    'scale_variation': [6458, 3794, 1982],
    'truncation': [7050, 2166, 2166, 852],
    'occlusion': [8166, 1806, 1806, 456],
}
SOURCES = {
    'paper': 'https://arxiv.org/abs/1908.10543',
    'official_evaluator': 'https://github.com/juhongm999/hpf/blob/master/model/evaluation.py',
    'official_dataset': 'https://github.com/juhongm999/hpf/blob/master/data/spair.py',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_annotations(annotations, images):
    assert len(annotations) == 12234
    assert len({a['filename'] for a in annotations}) == len(annotations)
    counts = {}
    for factor, labels in FACTORS.items():
        counter = Counter(a[factor] for a in annotations)
        counts[factor] = [counter[i] for i in range(len(labels))]
        assert counts[factor] == EXPECTED[factor], (factor, counter)
        assert set(counter) == set(range(len(labels)))
    for a in annotations:
        cls = images[a['category']]
        src, tgt = [cls[a[k + '_imname'][:-4]] for k in ('src', 'trg')]
        for factor, key in [('truncation', 'truncated'), ('occlusion', 'occluded')]:
            assert a[factor] == int(src[key]) + 2 * int(tgt[key]), (a['filename'], factor)
        common = sorted(set(i for i, xy in src['kps'].items() if xy) &
                        set(i for i, xy in tgt['kps'].items() if xy), key=int)
        assert common == a['kps_ids'], a['filename']
        for k, image in [('src', src), ('trg', tgt)]:
            assert [image['kps'][i] for i in common] == a[k + '_kps'], a['filename']
        x1, y1, x2, y2 = a['trg_bndbox']
        assert max(x2 - x1, y2 - y1) > 0
    return counts


def prepare(args):
    root, crisp = args.root.resolve(), args.crisp_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    originals = [crisp / 'evaluate_spair_correspondence.py', crisp / 'LICENSE']
    for directory in ('evals', 'configs'):
        originals += [p for p in (crisp / directory).rglob('*')
                      if p.is_file() and p.suffix in ('.py', '.yaml') and '__pycache__' not in p.parts]
    hashes = {}
    for p in originals:
        relative = p.relative_to(crisp)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(relative)] = digest(target)
    shutil.copy2(Path(__file__), root / 'spair_nuisance.py')
    data = crisp / 'data/SPair-71k'
    layout = data / 'Layout/large/test.txt'
    ids = layout.read_text().splitlines()
    assert len(ids) == 12234 and len(set(ids)) == 12234
    images, annotations = {}, []
    for i, pair in enumerate(ids):
        a = json.loads((data / 'PairAnnotation/test' / (pair + '.json')).read_text())
        assert a['filename'] == pair
        annotations.append(a)
        category = images.setdefault(a['category'], {})
        for key in ('src', 'trg'):
            image_id = a[key + '_imname'][:-4]
            if image_id not in category:
                category[image_id] = json.loads((data / 'ImageAnnotation' / a['category'] / (image_id + '.json')).read_text())
                assert (data / 'JPEGImages' / a['category'] / (image_id + '.jpg')).is_file()
                assert (data / 'Segmentation' / a['category'] / (image_id + '.png')).is_file()
        if (i + 1) % 2000 == 0:
            print('Validating official annotations', i + 1, '/', len(ids), flush=True)
    counts = validate_annotations(annotations, images)
    save(root / 'annotations.json', {'pairs': annotations, 'images': images})
    checkpoints = {}
    for name, path in MODELS.items():
        print('Hashing checkpoint', name, flush=True)
        checkpoints[name] = {'path': str(path), 'sha256': digest(path)}
        (root / name).mkdir()
    save(root / 'manifest.json', {
        'crisp_root': str(crisp), 'data_root': str(data.resolve()), 'checkpoints': checkpoints,
        'source_sha256': hashes, 'runner_sha256': digest(root / 'spair_nuisance.py'),
        'annotations_sha256': digest(root / 'annotations.json'), 'layout_sha256': digest(layout),
        'subset_counts': counts, 'test_pairs': len(ids), 'sources': SOURCES,
        'protocol': {'features': 'raw final block 12 before final LayerNorm; no projection head',
                     'matching': 'unmodified supplied CRISP compute_errors; L2 then bilinear source sampling and cosine argmax',
                     'image_size': 800, 'bbox_crop': False, 'image_mean': 'imagenet',
                     'pck_alpha_bbox': 0.1, 'comparison': '<= (official HPF)',
                     'aggregation': 'mean of per-pair PCK (official HPF)', 'test_time_augmentation': False,
                     'secondary_aggregation': 'keypoint micro average and Probe3D category macro average',
                     'subset_selection': 'each factor independently, other factors unrestricted (paper Table 4)',
                     'preprocessing': 'native Probe3D square padding and integer keypoint rescaling retained'},
    })
    collect(args)
    print('Prepared', root, flush=True)


def summarize(rows):
    def score(selected):
        if not selected:
            return None
        correct, total = sum(r['correct'] for r in selected), sum(r['keypoints'] for r in selected)
        classes = sorted({r['category'] for r in selected})
        macro = sum(sum(r['correct'] for r in selected if r['category'] == c) /
                    sum(r['keypoints'] for r in selected if r['category'] == c) for c in classes) / len(classes)
        return {'pck': 100 * sum(r['correct'] / r['keypoints'] for r in selected) / len(selected),
                'keypoint_micro_pck': 100 * correct / total, 'probe3d_category_macro_pck': 100 * macro,
                'pairs': len(selected), 'keypoints': total, 'correct_keypoints': correct}
    return {'all': score(rows), 'subsets': {
        factor: {label: score([r for r in rows if r[factor] == i]) for i, label in enumerate(labels)}
        for factor, labels in FACTORS.items()}}


def run(args):
    import torch
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    source = root / 'source'
    sys.path.insert(0, str(source))
    os.chdir(source)
    from evals.datasets.spair import SPairDataset
    from evals.models.region_vits import RegionViTS
    from evaluate_spair_correspondence import compute_errors
    for relative, expected in manifest['source_sha256'].items():
        assert digest(source / relative) == expected, relative
    assert digest(root / 'spair_nuisance.py') == manifest['runner_sha256']
    assert digest(root / 'annotations.json') == manifest['annotations_sha256']
    identity = manifest['checkpoints'][args.model]
    assert digest(identity['path']) == identity['sha256']
    data = json.loads((root / 'annotations.json').read_text())
    assert validate_annotations(data['pairs'], data['images']) == manifest['subset_counts']

    class OrderedSPair(SPairDataset):
        def get_pair_annotations(self):
            return data['pairs']

        def get_image_annotations(self):
            return data['images']

    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    model = RegionViTS(identity['path'], output='dense', return_multilayer=False).cuda().eval()
    model.checkpoint_name = args.model
    assert model.multilayers == [11] and model.feat_dim == 384
    assert not any(p.requires_grad for p in model.parameters())
    # Same strict adapter is valid for both original iBOT and continuation teachers.
    checkpoint = torch.load(identity['path'], map_location='cpu')
    epoch = checkpoint.get('epoch')
    assert epoch == (200 if args.model == 'region200' else 800), epoch
    del checkpoint
    dataset = OrderedSPair(manifest['data_root'], 'test', image_size=800,
                           image_mean='imagenet', use_bbox=False, num_instances=None)
    out = root / args.model
    journal = out / 'pairs.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    for i, row in enumerate(rows):
        assert row['filename'] == data['pairs'][i]['filename']
    assert len(rows) <= len(dataset)
    started = time.monotonic()
    with torch.inference_mode():
        # Validate the real matching path on source-only and target-only conditions.
        smoke = [0] + [next(i for i, a in enumerate(data['pairs']) if a[f] == v)
                       for f in ('truncation', 'occlusion') for v in (1, 2)]
        for i in smoke:
            errors = compute_errors(model, dataset[i])[0]
            assert errors.numel() == len(data['pairs'][i]['kps_ids'])
            assert torch.isfinite(errors).all()
        save(out / 'preflight.json', {'passed': True, 'epoch': epoch, 'checkpoint': identity,
                                    'subset_counts': manifest['subset_counts'], 'sample_indices': smoke,
                                    'gpu': torch.cuda.get_device_name(),
                                    'peak_gpu_memory_gib': torch.cuda.max_memory_allocated() / 1024**3})
        print('PREFLIGHT PASS', args.model, 'starting at pair', len(rows), flush=True)
        with journal.open('a', buffering=1) as handle:
            for i in range(len(rows), len(dataset)):
                a = data['pairs'][i]
                errors = compute_errors(model, dataset[i])[0]
                assert errors.numel() == len(a['kps_ids']) and errors.numel() > 0, a['filename']
                assert torch.isfinite(errors).all(), a['filename']
                row = {k: a[k] for k in ['filename', 'category'] + list(FACTORS)}
                row.update(correct=int((errors <= 0.1).sum()), keypoints=errors.numel(),
                           normalized_errors=errors.tolist())
                handle.write(json.dumps(row) + '\n')
                rows.append(row)
                if (i + 1) % 100 == 0:
                    print(args.model, 'pairs', i + 1, '/', len(dataset),
                          'elapsed_minutes', round((time.monotonic() - started) / 60, 1), flush=True)
    save(out / 'results.json', {'status': 'completed', 'model': args.model, 'checkpoint': identity,
                              'protocol': manifest['protocol'], 'sources': SOURCES,
                              'epoch': epoch, 'elapsed_seconds_this_invocation': time.monotonic() - started,
                              'metrics': summarize(rows)})
    print('COMPLETED', args.model, flush=True)


def collect(args):
    root = args.root.resolve()
    results = {}
    for model in MODELS:
        path = root / model / 'results.json'
        results[model] = json.loads(path.read_text()) if path.exists() else {'status': 'not_finished'}
    save(root / 'results.json', results)
    cols = [(factor, label) for factor, labels in FACTORS.items() for label in labels]
    lines = ['# SPair-71k nuisance PCK@0.1 (official per-pair average)', '',
             '| Model | ' + ' | '.join(f'{f}/{l}' for f, l in cols) + ' |',
             '|---|' + '---:|' * len(cols)]
    tex_rows = []
    for model, label in [('ibot_original', r'iBOT~\cite{zhou2022ibot}'), ('region200', r'\rowcolor{blue}\textbf{Ours}')]:
        entry = results[model]
        scores = [f"{entry['metrics']['subsets'][f][l]['pck']:.2f}" if entry['status'] == 'completed' else '--' for f, l in cols]
        lines.append('| ' + model + ' | ' + ' | '.join(scores) + ' |')
        tex_rows.append(label + ' & ' + ' & '.join(scores) + r' \\')
    (root / 'results_summary.md').write_text('\n'.join(lines) + '\n')
    (root / 'table_rows.tex').write_text('\n'.join(tex_rows) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'collect'])
    p.add_argument('--root', type=Path, default=PROJECT / 'output/analysis' / ('spair_nuisance_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--crisp-root', type=Path, default=CRISP)
    p.add_argument('--model', choices=list(MODELS))
    args = p.parse_args()
    if args.action == 'run' and args.model is None:
        p.error('--model is required for run')
    globals()[args.action](args)


if __name__ == '__main__':
    main()
