"""Prepare and run the supplied CRISP depth/surface-normal benchmarks.

Uses the original loaders, DPT heads, training loops, losses and metrics.
Run with the supplied probe3d environment, not the main training environment.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
from datetime import datetime

CRISP_ROOT = Path('/mnt/fast/nobackup/scratch4weeks/jg02228/probe3d')
CHECKPOINT = Path('/mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/output/long_ibot_vit_small/85535_0/checkpoint_source1000_continuation0200.pth')
TASKS = {
    'depth_nyu': ('depth', 'nyu', 2),
    'depth_navi': ('depth', 'navi_reldepth', 2),
    'snorm_nyu': ('snorm', 'nyu', 8),
    'snorm_navi': ('snorm', 'navi_reldepth', 8),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    crisp = args.crisp_root.resolve()
    checkpoint = args.checkpoint.resolve()
    assert checkpoint.is_file(), checkpoint
    for relative in ('data/nyu_geonet', 'data/nyuv2/nyuv2_snorm_all.pkl', 'data/navi_v1'):
        assert (crisp / relative).exists(), relative
    assert (crisp / 'env_probe3d/bin/python').is_file()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    originals = [crisp / 'train_depth.py', crisp / 'train_snorm.py', crisp / 'LICENSE']
    for directory in ('evals', 'configs'):
        originals.extend(p for p in (crisp / directory).rglob('*')
                         if p.is_file() and p.suffix in ('.py', '.yaml')
                         and '__pycache__' not in p.parts)
    source = root / 'source'
    hashes = {}
    for original in originals:
        relative = original.relative_to(crisp)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        hashes[str(relative)] = digest(original)
    adapter = Path(__file__).parent / 'utils/crisp_checkpoint_features.py'
    shutil.copy2(adapter, source / 'evals/models/checkpoint_vits.py')
    # The native single-GPU path has no .module wrapper. This changes only
    # serialization after final evaluation; training and metrics are untouched.
    for entry in ('train_depth.py', 'train_snorm.py'):
        path = source / entry
        text = path.read_text()
        for module in ('model', 'probe'):
            old = f'"{module}": {module}.module.state_dict(),'
            new = f'"{module}": getattr({module}, "module", {module}).state_dict(),'
            assert text.count(old) == 1, (entry, old)
            text = text.replace(old, new)
        path.write_text(text)
    backbone = source / 'configs/backbone/region_vits200.yaml'
    backbone.write_text('_target_: evals.models.checkpoint_vits.CheckpointViTS\n'
                        f'checkpoint_path: {checkpoint}\noutput: dense\nlayer: -1\n'
                        'return_multilayer: true\n' + f'checkpoint_name: {args.model_name}\n')
    shutil.copy2(Path(__file__), source / 'native_dense.py')
    tasks = {}
    for name, (kind, dataset, batch) in TASKS.items():
        if name not in args.tasks:
            continue
        task_root = root / 'tasks' / name
        shutil.copytree(source, task_root / 'source')
        (task_root / 'source/data').symlink_to(crisp / 'data', target_is_directory=True)
        tasks[name] = {'kind': kind, 'dataset': dataset, 'batch_size': batch,
                       'num_gpus': 1, 'task_root': str(task_root)}
    manifest = {
        'crisp_root': str(crisp), 'checkpoint': str(checkpoint), 'model_name': args.model_name,
        'checkpoint_sha256': digest(checkpoint), 'original_source_sha256': hashes,
        'snapshot_source_sha256': {str(p.relative_to(source)): digest(p)
                                  for p in source.rglob('*') if p.is_file()},
        'changes': ['CheckpointViTS adapter strictly loads the selected teacher ViT-S; Crisper.forward unchanged',
                    'Depth and surface-normal checkpoint saving unwraps DDP only when present',
                    'Local output paths and single-GPU resources; saved CRISP batch sizes preserved'],
        'features': 'raw blocks 3/6/9/12, separate DPT inputs, no final LayerNorm/head/scaler',
        'tasks': tasks, 'environment': str(crisp / 'env_probe3d/bin/python'),
        'protocol': {'epochs': 10, 'warmup_epochs': 1.5, 'probe_lr': 0.0005,
                     'model_lr': 0.0, 'precision': 'native FP32',
                     'nyu': 'GeoNet trainval (30914), labeled test (654), 480x480 center crop',
                     'navi': 'multiview all trainval, wild all test, original every-fourth selection, 512x512'},
    }
    write_json(root / 'manifest.json', manifest)
    shutil.copy2(Path(__file__), root / 'native_dense.py')
    (root / 'README.md').write_text(
        '# Native CRISP depth and surface normals: region ViT-S epoch 200\n\n'
        'Four independent single-GPU benchmarks. See manifest.json for code hashes, '
        'checkpoint identity, exact settings and data locations. Each task performs a '
        'data/model/decoder preflight before executing the original training entry point. '
        'No benchmark subsampling or feature normalization is added.\n\n'
        'Results: `python native_dense.py collect --root .`\n'
        'Training logs and final decoder checkpoints are under each '
        '`tasks/*/source/{depth,snorm}_exps/` directory.\n'
        'The supplied code declares random_seed=8 but does not seed its training '
        'entry points; this behavior is preserved.\n')
    print(root, flush=True)


def context(root, task):
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['tasks'][task]
    source = Path(config['task_root']) / 'source'
    os.chdir(source)
    sys.path.insert(0, str(source))
    return manifest, config, source


def preflight(args):
    import torch
    import torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    manifest, task, source = context(args.root.resolve(), args.task)
    from evals.utils.losses import DepthLoss, angular_loss
    from evals.utils.metrics import evaluate_depth, evaluate_surface_norm
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    for relative, expected in manifest['snapshot_source_sha256'].items():
        assert digest(source / relative) == expected, relative
    assert digest(manifest['checkpoint']) == manifest['checkpoint_sha256']
    with initialize_config_dir(config_dir=str(source / 'configs'), version_base=None):
        cfg = compose(config_name=task['kind'] + '_training', overrides=[
            'backbone=region_vits200', 'dataset=' + task['dataset'],
            'system.num_gpus=1', 'batch_size=' + str(task['batch_size'])])
    assert cfg.optimizer.model_lr == 0 and cfg.optimizer.n_epochs == 10
    assert cfg.optimizer.probe_lr == 0.0005 and cfg.optimizer.warmup_epochs == 1.5
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu')
    assert 'teacher' in checkpoint, 'Teacher state is required'
    checkpoint_epoch = checkpoint.get('epoch')
    del checkpoint
    gc.collect()
    report = {'task': args.task, 'data_root': str((source / 'data').resolve()),
              'checkpoint_epoch': checkpoint_epoch,
              'checkpoint_sha256': manifest['checkpoint_sha256'], 'splits': {},
              'config': __import__('omegaconf').OmegaConf.to_container(cfg, resolve=True)}
    train_sample = None
    for split in ('trainval', 'test'):
        dataset = instantiate(cfg.dataset, split=split)
        identities = getattr(dataset, 'instances', getattr(dataset, 'indices', None))
        identities = list(identities)
        write_json(Path(task['task_root']) / (split + '_instances.json'),
                   [x.item() if hasattr(x, 'item') else x for x in identities])
        if task['dataset'] == 'nyu':
            assert len(dataset) == (30914 if split == 'trainval' else 654)
            if split == 'trainval':
                for identity in identities:
                    assert (Path(dataset.root_dir) / identity).is_file(), identity
        else:
            assert len(dataset) == (2024 if split == 'trainval' else 555)
            for obj, collection, image in identities:
                for directory, ext in [('images', '.jpg'), ('depth', '.png')]:
                    file = Path(dataset.data_root) / obj / collection / directory / ('downsampled_' + image + ext)
                    assert file.is_file(), file
        samples = []
        for i in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[i]
            side = 480 if task['dataset'] == 'nyu' else 512
            assert sample['image'].shape == (3, side, side)
            assert sample['depth'].shape == (1, side, side)
            assert sample['snorm'].shape == (3, side, side)
            for key in ('image', 'depth', 'snorm'):
                assert torch.isfinite(sample[key]).all(), (split, i, key)
            assert (sample['depth'] > 0).any(), (split, i)
            samples.append(i)
            if split == 'trainval' and i == 0:
                train_sample = {key: sample[key].clone() for key in ('image', 'depth', 'snorm')}
        report['splits'][split] = {'count': len(dataset), 'sample_indices_checked': samples,
                                  'instances_sha256': digest(Path(task['task_root']) / (split + '_instances.json'))}
        print('DATA PASS', args.task, split, len(dataset), flush=True)
        del dataset, sample
        gc.collect()
    model = instantiate(cfg.backbone).cuda()
    assert model.multilayers == [2, 5, 8, 11]
    assert model.feat_dim == [384] * 4
    assert not any(p.requires_grad for p in model.parameters())
    batch = task['batch_size']
    images = train_sample['image'].unsqueeze(0).repeat(batch, 1, 1, 1).cuda()
    target = train_sample['depth' if task['kind'] == 'depth' else 'snorm'].unsqueeze(0).repeat(batch, 1, 1, 1).cuda()
    with torch.no_grad():
        features = model(images)
    report['feature_shapes'] = [list(x.shape) for x in features]
    assert all(torch.isfinite(x).all() for x in features)
    if task['kind'] == 'depth':
        probe = instantiate(cfg.probe, feat_dim=model.feat_dim, max_depth=10.0 if task['dataset'] == 'nyu' else 1.0).cuda()
        pred = F.interpolate(probe(features), size=target.shape[-2:], mode='bilinear', align_corners=True)
        loss = DepthLoss()(pred, target)
        metrics = evaluate_depth(pred.detach(), target)
    else:
        probe = instantiate(cfg.probe, feat_dim=model.feat_dim).cuda()
        pred = F.interpolate(probe(features), size=target.shape[-2:], mode='bicubic', align_corners=True)
        mask = train_sample['depth'].unsqueeze(0).repeat(batch, 1, 1, 1).cuda() > 0
        loss = angular_loss(pred, target, mask, uncertainty_aware=True)
        metrics = evaluate_surface_norm(pred.detach(), target, mask)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in probe.parameters() if p.grad is not None)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    report['smoke_loss'] = loss.item()
    report['peak_gpu_memory_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
    report['gpu'] = torch.cuda.get_device_name()
    report['passed'] = True
    write_json(Path(task['task_root']) / 'preflight_manifest.json', report)
    print('PREFLIGHT PASS', args.task, report['peak_gpu_memory_gib'], 'GiB', flush=True)


def run(args):
    _, task, source = context(args.root.resolve(), args.task)
    assert json.loads((Path(task['task_root']) / 'preflight_manifest.json').read_text())['passed']
    import torch
    torch.set_num_threads(4)
    sys.argv = [str(source / ('train_' + task['kind'] + '.py')),
                'backbone=region_vits200', 'dataset=' + task['dataset'],
                'system.num_gpus=1', 'batch_size=' + str(task['batch_size']),
                'note=' + args.task,
                'hydra.run.dir=' + str(Path(task['task_root']) / 'hydra'),
                'hydra.job.chdir=false']
    runpy.run_path(sys.argv[0], run_name='__main__')


def collect(args):
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    results = {}
    for name, task in manifest['tasks'].items():
        source = Path(task['task_root']) / 'source'
        logs = sorted((source / (task['kind'] + '_exps')).glob('*/training.log'))
        entry = {'status': 'not_finished', 'metrics': {}}
        if logs:
            log = logs[-1]
            text = log.read_text()
            pattern = r'Final test (SA |SI )?(d[123]|rmse)\s*\|\s*([0-9.eE+-]+)'
            for scale, metric, value in re.findall(pattern, text):
                entry['metrics'].setdefault(scale.strip() or 'normal', {})[metric] = float(value)
            entry['training_log'] = str(log)
            if 'Saved checkpoint at' in text and (log.parent / 'ckpt.pth').is_file():
                entry['status'] = 'completed'
                entry['checkpoint'] = str(log.parent / 'ckpt.pth')
        results[name] = entry
    write_json(root / 'results.json', results)
    lines = ['# Native CRISP dense benchmarks: ' + manifest.get('model_name', 'region ViT-S epoch 200'), '',
             '| Benchmark | Status | Scale | d1 | d2 | d3 | RMSE |',
             '|---|---|---|---:|---:|---:|---:|']
    for name, entry in results.items():
        for scale, metrics in (entry['metrics'] or {'--': {}}).items():
            values = [f'{metrics[key]:.4f}' if key in metrics else '--' for key in ('d1', 'd2', 'd3', 'rmse')]
            lines.append('| ' + ' | '.join([name, entry['status'], scale] + values) + ' |')
    (root / 'results_summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'run', 'collect'))
    parser.add_argument('--root', type=Path, default=Path('output/analysis') / ('crisp_native_dense_region200_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--crisp-root', type=Path, default=CRISP_ROOT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--model-name', default='ibot_region200_vits16')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    if args.action in ('preflight', 'run'):
        parser.error('--task is required') if args.task is None else None
    globals()[args.action](args)


if __name__ == '__main__':
    main()
