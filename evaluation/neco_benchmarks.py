"""Thin, audited adapters around pinned NeCo and Open Hummingbird evaluations.

No projection head or softmax features. See neco_protocol.md for code/paper
discrepancies and explicitly recorded compatibility changes.
"""
import argparse
import importlib
import json
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / 'evaluation/vendor'


def prepare(root, config):
    """Freeze a reproducible comparison, including all published subset seeds."""
    root.mkdir(parents=True, exist_ok=False)
    tasks = []
    def append(evaluation, dataset, **options):
        for model, checkpoint in config['models'].items():
            checkpoint = Path(checkpoint).resolve(strict=True)
            tasks.append(dict(evaluation=evaluation, dataset=dataset, model=model,
                              checkpoint=str(checkpoint), **options))
    evaluations = config.get('evaluations', ['clustering', 'linear', 'retrieval'])
    if 'clustering' in evaluations:
        for dataset, classes in [('voc', 21), ('ade20k', 151), ('coco-thing', 12), ('coco-stuff', 15)]:
            for k in config.get('clustering', {}).get('clusters', [classes, 500]):
                append('clustering', dataset, clusters=k)
    if 'fully_unsupervised' in evaluations:
        append('fully_unsupervised', 'voc')
    if 'linear' in evaluations:
        for dataset in ('coco-thing', 'coco-stuff'):
            append('linear', dataset)
    if 'retrieval' in evaluations:
        for fraction in (128, 64, 8, 1):
            for dataset in ('voc', 'ade20k'):
                for seed in ([42] if fraction == 1 else config['retrieval']['subset_seeds']):
                    append('retrieval', dataset, fraction=fraction, seed=seed)
    save(root / 'config.yaml', config)
    save(root / 'tasks.json', tasks)
    for folder in ('evaluation', 'model', 'utils'):
        shutil.copytree(PROJECT / folder, root / 'source' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    print('Prepared', len(tasks), 'tasks in', root, flush=True)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def strict_neco_model(checkpoint):
    import torch
    from src.models.vit import vit_small
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {}
    for name, tensor in payload['teacher'].items():
        while name.startswith(('module.', '_orig_mod.')):
            name = name.split('.', 1)[1]
        if name.startswith('backbone.'):
            state[name[len('backbone.'):]] = tensor
    model = vit_small(patch_size=16)
    keys = {k for k in model.state_dict() if not k.startswith('projection_head.')}
    missing = keys - state.keys()
    unexpected = state.keys() - keys - {'masked_embed', 'norm_cls.weight', 'norm_cls.bias'}
    if missing or unexpected:
        raise ValueError(f'Incomplete teacher: missing={missing}, unexpected={unexpected}')
    state = {k: state[k] for k in keys}
    msg = model.load_state_dict(state, strict=False)
    assert all(k.startswith('projection_head.') for k in msg.missing_keys), msg
    assert not msg.unexpected_keys
    model.eval().requires_grad_(False)
    return model, state, {'checkpoint': str(Path(checkpoint).resolve()),
                         'checkpoint_epoch': payload.get('epoch'),
                         'teacher_backbone_tensors_loaded': len(state)}


def run_retrieval(task, out, config):
    import numpy as np
    import torch
    sys.path.insert(0, str(VENDOR / 'hummingbird'))
    from hbird import hbird_eval as module
    model, _, metadata = strict_neco_model(task['checkpoint'])
    seed = task['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    subsets = VENDOR / 'hummingbird/file_sets' / task['dataset']
    stem = 'trainaug' if task['dataset'] == 'voc' else 'training'
    train = subsets / (f'1_div_{task["fraction"]}/{stem}_{task["fraction"]}_{seed}.txt'
                       if task['fraction'] > 1 else f'full/{stem}.txt')
    # ADE full list isn't included in the release; the loader's official full
    # split is used, as in upstream. Partial lists are always released lists.
    if task['fraction'] == 1 and not train.exists():
        train = None
    n = len(train.read_text().split()) if train else 20210
    memory = config['retrieval']['memory_size']
    input_size = config['retrieval']['input_size']
    # Same fixed memory capacity and feature sampler at every fraction. Enough
    # augmentation passes to avoid topk(k > patches) in the released sampler.
    epochs = max(2, math.ceil(memory / (n * (input_size // 16) ** 2)))
    details = []
    original_create = module.HbirdEvaluation.create_memory
    def create_memory(self, loader, *args, **kwargs):
        original_create(self, loader, *args, **kwargs)
        filled = n * epochs * self.num_sampled_features
        self.feature_memory = self.feature_memory[:filled]
        self.label_memory = self.label_memory[:filled]
        details.append({'capacity': memory, 'populated_entries': filled, 'augmentation_passes': epochs})
    module.HbirdEvaluation.create_memory = create_memory
    # Retain the final incomplete batch. Otherwise VOC silently drops images
    # from both the support set and the scored validation set.
    original_voc = module.VOCDataModule
    class AllVOC(original_voc):
        def __init__(self, *args, **kwargs):
            kwargs['drop_last'] = False
            super().__init__(*args, **kwargs)
    module.VOCDataModule = AllVOC
    # ScaNN's TF build must not reserve the GPU needed by the frozen backbone.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    root = Path(config['data_root'])
    data = root / 'voc' if task['dataset'] == 'voc' else Path(config['ade20k'])
    score = module.hbird_evaluation(
        model.cuda(), d_model=384, patch_size=16, dataset_name=task['dataset'],
        data_dir=str(data), batch_size=16, input_size=input_size,
        augmentation_epoch=epochs, device='cuda', n_neighbours=30,
        nn_method='scann', memory_size=memory, num_workers=4,
        ftr_extr_fn=lambda m, x: (m.forward_backbone(x)[:, 1:], None),
        train_fs_path=str(train) if train else None,
        val_fs_path=str(root / 'voc/sets/val.txt') if task['dataset'] == 'voc' else None)
    return {'miou_percent': float(score) * 100, 'checkpoint': metadata,
            'memory': details, 'support_images': n, 'seed': seed,
            'support_list': str(train) if train else 'official complete ADE20K training split',
            'support_list_sha256': hashlib.sha256(train.read_bytes()).hexdigest() if train else None}


def run_native(task, out, config):
    import torch
    import yaml
    from pytorch_lightning.loggers import CSVLogger
    from experiments import utils
    # Map upstream logger writes to local CSV files, never to an external app.
    class ExperimentProxy:
        def __init__(self, writer):
            self.writer = writer
        def __getattr__(self, name):
            return getattr(self.writer, name)
        def __setitem__(self, key, value):
            self.writer.log_hparams({key: value})
    class LocalCSV(CSVLogger):
        @property
        def experiment(self):
            return ExperimentProxy(super().experiment)
    scores = []
    original_compute = utils.PredsmIoU.compute
    def compute(self, *args, **kwargs):
        result = original_compute(self, *args, **kwargs)
        if result is not None:
            scores.append({'miou_percent': float(result[0]) * 100,
                           'tp': [int(x) for x in result[1]],
                           'fp': [int(x) for x in result[2]],
                           'fn': [int(x) for x in result[3]]})
            save(out / 'metrics_so_far.json', scores)
        return result
    utils.PredsmIoU.compute = compute
    _, state, metadata = strict_neco_model(task['checkpoint'])
    def weights(arch, method, patch_size=None, weight_prefix='model', **kwargs):
        return {(weight_prefix + '.' if weight_prefix else '') + k: v for k, v in state.items()}
    utils.get_backbone_weights = weights
    linear = task['evaluation'] == 'linear'
    module = importlib.import_module('experiments.linear_segmentation.linear_finetune' if linear
                                    else 'experiments.overcluster.eval_overcluster')
    module.get_backbone_weights = weights
    module.NeptuneLogger = lambda **kwargs: LocalCSV(str(out), name='local_metrics')
    if task['dataset'].startswith('coco'):
        original_coco = module.CocoDataModule
        class AuditedCoco(original_coco):
            def __init__(self, *args, **kwargs):
                save(out / 'split_manifest.json', {
                    'seed': 400, 'training_files': kwargs['file_list'],
                    'validation_files': kwargs['file_list_val']})
                super().__init__(*args, **kwargs)
        module.CocoDataModule = AuditedCoco
    dataset = task['dataset']
    upstream = VENDOR / 'neco/experiments' / ('linear_segmentation' if linear else 'overcluster')
    confname = {'voc': 'pascal', 'ade20k': 'ade20k', 'coco-thing': 'coco-things', 'coco-stuff': 'coco-stuff'}[dataset]
    conf = yaml.safe_load((upstream / f'configs/{confname}/neco-dino.yml').read_text())
    conf['num_workers'] = 4
    conf['log_status'] = 'offline'
    conf['data']['data_dir'] = str(Path(config['ade20k']) if dataset == 'ade20k'
                                  else Path(config['data_root']) if dataset == 'voc'
                                  else Path(config['data_root']) / 'coco')
    opts = conf['train' if linear else 'val']
    opts.update(ckpt_path=task['checkpoint'], ckpt_dir=str(out / 'heads'), method='custom')
    if linear:
        opts['max_epochs'] = config['linear']['epochs']
    else:
        opts['K'] = task['clusters']
    config_path = out / 'upstream_config.yaml'
    config_path.write_text(yaml.safe_dump(conf, sort_keys=False))
    module.entry.main(args=['--config_path', str(config_path.resolve())], standalone_mode=False)
    if not scores:
        raise RuntimeError('Upstream run produced no evaluation metrics')
    result = {'checkpoint': metadata, 'scores': scores, 'config': conf}
    if linear:
        result['best_training_validation_miou_percent_100px'] = max(s['miou_percent'] for s in scores)
        # Run the released final evaluator at its default 448px mask resolution.
        from experiments.linear_segmentation import eval_linear
        eval_linear.get_backbone_weights = weights
        candidates = list((out / 'heads').rglob('*.ckpt'))
        if not candidates:
            raise RuntimeError('Missing trained linear-head checkpoint')
        def saved_score(p):
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            callbacks = ckpt.get('callbacks', {})
            return max(float(v.get('current_score', -1)) for v in callbacks.values() if isinstance(v, dict))
        best = max(candidates, key=saved_score)
        ckpt = torch.load(best, map_location='cpu', weights_only=False)
        head = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('finetune_head.')}
        head_path = out / 'linear_head.pth'
        torch.save(head, head_path)
        # PyTorch >=2.6 changed torch.load's default. This head contains tensors only.
        eval_linear.eval_bulk.callback(
            ckpt_path_backbone=task['checkpoint'], ckpt_path_head=str(head_path), patch_size=16,
            arch='vit-small', num_classes=12 if dataset == 'coco-thing' else 15,
            head_type='linear', dataset_name=dataset, data_dir=conf['data']['data_dir'],
            batch_size=15, input_size=448, mask_eval_size=448, arch_version='v1', num_register_tokens=0)
        result['final_miou_percent_448px'] = scores[-1]['miou_percent']
        result['selected_head'] = str(best)
    else:
        result['mean_miou_percent'] = sum(s['miou_percent'] for s in scores) / len(scores)
        assert len(scores) == 5, len(scores)
    return result


def main():
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', type=int)
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    if a.prepare:
        prepare(a.root.resolve(), config)
        return
    if a.task is None:
        p.error('--task is required unless --prepare is used')
    task = json.loads((a.root / 'tasks.json').read_text())[a.task]
    out = a.root / f'task_{a.task:03d}_{task["evaluation"]}_{task["dataset"]}_{task["model"]}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        print('Already complete:', out, flush=True)
        return
    data_needed = 'voc' if task['dataset'] == 'voc' else 'coco' if task['dataset'].startswith('coco') else None
    if data_needed and not (Path(config['data_root']) / f'{data_needed}_ready.json').is_file():
        raise RuntimeError('Data preparation audit is missing: ' + data_needed)
    sys.path.insert(0, str(VENDOR / 'neco'))
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU is required')
    save(out / 'identity.json', {'task': task, 'provenance': json.loads((VENDOR / 'neco_provenance.json').read_text())})
    if task['evaluation'] == 'fully_unsupervised':
        from evaluation.neco_fully_unsupervised import run_fully_unsupervised
        runner = run_fully_unsupervised
    else:
        runner = run_retrieval if task['evaluation'] == 'retrieval' else run_native
    result = runner(task, out, config)
    result.update(task=task, complete=True)
    save(out / 'results.json', result)


if __name__ == '__main__':
    main()
