"""Compatibility adapter for the pinned NeCo CBFE + community detection code.

The vendor source is left intact. Every runtime compatibility change is recorded
in results.json; see neco_protocol.md for code/paper differences.
"""
import ast
import hashlib
import importlib
import inspect
import importlib.metadata
from pathlib import Path
import shutil
import textwrap

from evaluation.neco_benchmarks import VENDOR, save, strict_neco_model


def load_upstream():
    # scikit-image 0.19 renamed greycomatrix to graycomatrix.
    texture = importlib.import_module('skimage.feature.texture')
    if not hasattr(texture, 'greycomatrix'):
        texture.greycomatrix = texture.graycomatrix
    module = importlib.import_module('experiments.fully_unsup_seg.fully_unsup_seg')
    # Preserve fractional normalized co-occurrences. Upstream's integer tensor
    # raises a dtype error with current PyTorch and cannot represent these values.
    source = textwrap.dedent(inspect.getsource(module.create_matrix))
    old = 'torch.zeros(num_clusters, num_clusters).int()'
    assert source.count(old) == 1
    source = source.replace(old, 'torch.zeros(num_clusters, num_clusters, dtype=torch.float64)')
    exec(compile(ast.parse(source), '<NeCo create_matrix float accumulator>', 'exec'), module.__dict__)
    return module


def run_fully_unsupervised(task, out, config):
    import numpy as np
    import torch
    from experiments import utils

    module = load_upstream()
    options = config['fully_unsupervised']
    model, state, metadata = strict_neco_model(task['checkpoint'])
    # Cache each model separately: shared attention filenames in the released
    # code must never cause one checkpoint to consume the other's masks.
    cache = Path(options['cache_root']) / out.parent.name / task['model']
    cache.mkdir(parents=True, exist_ok=True)
    module.vit_small = lambda **kwargs: model
    module.get_backbone_weights = lambda **kwargs: state
    original_data = module.VOCDataModule
    class LocalVOC(original_data):
        def __init__(self, *args, **kwargs):
            kwargs['num_workers'] = 4
            super().__init__(*args, **kwargs)
            save(out / 'split_manifest.json', {
                'train': (Path(config['data_root']) / 'voc/sets/train.txt').read_text().split(),
                'val': (Path(config['data_root']) / 'voc/sets/val.txt').read_text().split(),
                'shuffle': False, 'drop_last': False,
            })
    module.VOCDataModule = LocalVOC

    # The helper uses 0.65 but the paper specifies 0.70. Keep the released
    # averaging, Gaussian filter and connected-component removal unchanged.
    original_attentions = utils.process_and_store_attentions
    def store_attentions(attns, threshold, spatial_res, split, experiment_folder):
        original_attentions(attns, options['attention_mass'], spatial_res, split, experiment_folder)
    utils.process_and_store_attentions = store_attentions
    original_store = module.store_and_compute_features
    def store_features(*args, **kwargs):
        original_store(*args, **kwargs)
        experiment_folder = Path(args[5])
        gt_save_folder = Path(kwargs['gt_save_folder'])
        # Upstream writes masks in experiment_folder but reads save_folder.
        for split in ('train', 'val'):
            shutil.copyfile(experiment_folder / f'attn_{split}.pt', gt_save_folder / f'attn_{split}.pt')
    module.store_and_compute_features = store_features

    # Infomap rejects seed 0; ten independent runs use seeds 1 through 10.
    original_infomap = module.Infomap
    seeds = []
    def infomap(*args, **kwargs):
        kwargs['seed'] += 1
        seeds.append(kwargs['seed'])
        return original_infomap(*args, **kwargs)
    module.Infomap = infomap

    scores = []
    original_metric = utils.PredsmIoU.compute
    def metric(self, *args, **kwargs):
        result = original_metric(self, *args, **kwargs)
        if result is not None:
            scores.append(float(result[0]) * 100)
            save(out / 'metrics_so_far.json', scores)
        return result
    utils.PredsmIoU.compute = metric
    torch.manual_seed(0)
    np.random.seed(0)
    module.start_unsup_seg(
        patch_size=16, arch='vit-small', arch_version='v1',
        ckpt_path=task['checkpoint'], experiment_name=task['model'],
        batch_size=15, input_size=448, save_folder=str(cache),
        data_dir=config['data_root'], pca_dim=50,
        k_fg_extraction=200, clustering_eval_size=100, evaluate_cbfe=True,
        clustering_seed=0, num_objects_pvoc=20,
        k_community=options['k_community'], markov_time=options['markov_time'],
        weight_threshold=options['weight_threshold'], num_runs=10,
        compute_upper_bound=False, split_cd='val',
    )
    assert len(scores) == 15, scores
    cbfe, cd = scores[:5], scores[5:]
    vendor = Path(inspect.getfile(module))
    return {
        'checkpoint': metadata, 'cache': str(cache),
        'cbfe_seed_miou_percent': cbfe,
        'cbfe_mean_miou_percent': float(np.mean(cbfe)),
        'community_detection_seed_miou_percent': cd,
        'community_detection_mean_miou_percent': float(np.mean(cd)),
        'community_detection_std_miou_percent': float(np.std(cd)),
        'community_detection_best_miou_percent': max(cd),
        'community_detection_seeds': seeds,
        'dependency_versions': {name: importlib.metadata.version(name)
                                for name in ('infomap', 'optuna', 'torch', 'scikit-image')},
        'protocol': {
            'input_size': 448, 'mask_eval_size': 100, 'pca_dim': 50,
            'attention_mass': options['attention_mass'],
            'k_fg_extraction': 200, 'k_community': options['k_community'],
            'markov_time': options['markov_time'], 'weight_threshold': options['weight_threshold'],
            'parameters': 'NeCo paper fixed parameters, identical for both checkpoints; no model-specific sweep',
            'cbfe_threshold_selection': 'released code uses training foreground ground truth',
            'feature_scaler_pca_and_clustering': 'released code combines train and validation features',
            'community_graph': 'released code combines train and validation co-occurrences',
            'vendor_source_sha256': hashlib.sha256(vendor.read_bytes()).hexdigest(),
            'compatibility_changes': [
                'greycomatrix import alias for current scikit-image',
                'float64 accumulator for fractional co-occurrences',
                'attention files copied to the directory read by CBFE',
                'Infomap seeds 1..10 instead of invalid zero seed',
                'four dataloader workers and strict frozen teacher checkpoint adapter',
            ],
            'paper_code_difference': 'paper 70% attention mass replaces helper 65%; fixed paper CD parameters',
        },
    }
