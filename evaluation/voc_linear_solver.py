"""Converged, regularized frozen linear VOC probes using existing image features."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from evaluation.utils.classification import multilabel_metrics
from evaluation.utils.classification_data import sample_few_shot_indices
from evaluation.voc_protocol_sweep import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--loss', choices=['bce', 'squared_hinge'], default='bce')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cached = torch.load(args.cache, map_location='cpu', weights_only=False)
    # Keep the current production feature point and representation fixed.
    x = cached['train']['features'][:, 1152:].cuda()
    xf = cached['train']['flipped'][:, 1152:].cuda()
    y = cached['train']['targets'].cuda()
    vx = cached['val']['features'][:, 1152:].cuda()
    vy = cached['val']['targets'].numpy()
    classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    result = {'status': 'running', 'cache_signature': cached['signature'], 'pooling': 'final-LayerNorm CLS + mean final patch tokens', 'solver': 'Full-batch L-BFGS, strong-Wolfe line search, max 500 iterations, gradient tolerance 1e-7, change tolerance 1e-9', 'loss': 'known-label mean BCE + 0.5 * lambda * sum(weight squared); bias unpenalized', 'caveat': 'Exploratory protocol comparison. Frozen features and labels unchanged. Solver convergence replaces the epoch-based SGD budget; not claimed CRISP equivalent.', 'results': []}
    if args.loss == 'squared_hinge':
        result['loss'] = 'Known-label mean squared hinge + 0.5 * lambda * sum(weight squared); one linear SVM per class, bias unpenalized'
    start = time.monotonic()
    for regime in ['1shot', 'full']:
        indices = sample_few_shot_indices(cached['train']['targets'].numpy(), 1, 0)[0] if regime == '1shot' else list(range(len(x)))
        xx = torch.cat([x[indices], xf[indices]])
        yy = y[indices].repeat(2, 1)
        known, target = yy >= 0, yy.clamp_min(0)
        denominator = known.sum()
        normalizations = ['raw', 'standard', 'l2'] if args.loss == 'squared_hinge' else ['raw', 'standard']
        for normalization in normalizations:
            standardize = normalization == 'standard'
            tx, validation = xx, vx
            if standardize:
                mean = xx.mean(0)
                std = xx.std(0, correction=0).clamp_min(1e-6)
                tx, validation = (xx - mean) / std, (vx - mean) / std
            elif normalization == 'l2':
                tx, validation = F.normalize(xx, dim=1), F.normalize(vx, dim=1)
            for regularization in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                weight = torch.zeros(20, tx.shape[1], device='cuda', requires_grad=True)
                bias = torch.zeros(20, device='cuda', requires_grad=True)
                optimizer = torch.optim.LBFGS([weight, bias], lr=1, max_iter=500, history_size=20, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')
                calls = 0
                def closure():
                    nonlocal calls
                    calls += 1
                    optimizer.zero_grad()
                    logits = F.linear(tx, weight, bias)
                    losses = F.binary_cross_entropy_with_logits(logits, target, reduction='none') if args.loss == 'bce' else (1 - (2 * target - 1) * logits).clamp_min(0).square()
                    loss = (losses * known).sum() / denominator
                    loss = loss + .5 * regularization * weight.square().sum()
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite loss')
                    loss.backward()
                    return loss
                optimizer.step(closure)
                final_loss = float(closure().detach())
                with torch.no_grad():
                    scores = F.linear(validation, weight, bias).cpu().numpy()
                metrics = multilabel_metrics(vy, scores, classes, 'pascal_voc')
                row = {'regime': regime, 'normalization': normalization, 'standardize': standardize, 'regularization': regularization, 'map_percent': metrics['map_percent'], 'average_precision_by_class': metrics['average_precision_by_class'], 'closure_calls': calls, 'iterations': optimizer.state[weight]['n_iter'], 'final_objective': final_loss, 'max_gradient': float(torch.cat([weight.grad.flatten(), bias.grad]).abs().max())}
                result['results'].append(row)
                save(args.output / 'results.json', result)
                print(json.dumps({k:v for k,v in row.items() if k != 'average_precision_by_class'}), flush=True)
    result['status'] = 'completed'
    result['elapsed_seconds'] = time.monotonic() - start
    save(args.output / 'results.json', result)


if __name__ == '__main__':
    main()
