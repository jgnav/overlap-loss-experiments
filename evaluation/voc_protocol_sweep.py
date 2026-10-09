"""Exploratory VOC control: frozen feature cache and parallel linear probes.

This is a diagnostic sweep, not a replacement benchmark. Whole-image resize
and flip have exactly two cached views; center-crop is a separate recipe.
Validation comparisons must not be presented as independent test results.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from evaluation.utils.common import load_backbone
from evaluation.utils.classification import multilabel_metrics
from evaluation.utils.classification_data import (
    read_multilabel_manifest, MultilabelDataset, sample_few_shot_indices,
)
from evaluation.utils.imagenet import IMAGENET_NORMALIZE


class AspectPreservingSquare:
    """Fit entire image inside a square; neutral ImageNet-mean padding."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = T.functional.resize(image, [new_height, new_width], interpolation=T.InterpolationMode.BICUBIC)
        left, top = (self.size - new_width) // 2, (self.size - new_height) // 2
        return T.functional.pad(image, [left, top, self.size - new_width - left, self.size - new_height - top], fill=[124, 116, 104])


class CropViews:
    def __init__(self, size, train_views=0, val_views=0, scale=(.08, 1.0), resize=256):
        self.train_views, self.val_views = train_views, val_views
        self.normalize = T.Compose([T.ToTensor(), IMAGENET_NORMALIZE])
        self.crop = T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC)
        self.flip = T.RandomHorizontalFlip()
        self.resize = T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC)
        self.multi = T.TenCrop(size) if val_views == 10 else T.FiveCrop(size)

    def __call__(self, image):
        if self.train_views:
            return torch.stack([self.normalize(self.flip(self.crop(image))) for _ in range(self.train_views)])
        return torch.stack([self.normalize(view) for view in self.multi(self.resize(image))])


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def voc2007_map(targets, scores):
    """Official VOC2007 eleven-point interpolated AP; ignore difficult labels."""
    aps = []
    for c in range(targets.shape[1]):
        known = targets[:, c] >= 0
        truth, prediction = targets[known, c], scores[known, c]
        positives = truth[np.argsort(-prediction, kind='stable')] == 1
        recall = np.cumsum(positives) / max(int(positives.sum()), 1)
        precision = np.cumsum(positives) / np.arange(1, len(positives) + 1)
        aps.append(np.mean([precision[recall >= threshold].max(initial=0) for threshold in np.linspace(0, 1, 11)]))
    return float(100 * np.mean(aps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--transform', choices=['square', 'center', 'pad'], default='square')
    p.add_argument('--resolution', type=int, default=224)
    p.add_argument('--pooling', nargs='+', choices=['cls_mean_patch', 'cls_last', 'cls_last4', 'cls_last4_mean_patch'])
    p.add_argument('--learning-rates', nargs='+', type=float)
    p.add_argument('--train-crops', type=int, default=0)
    p.add_argument('--crop-scale', nargs=2, type=float, default=[.08, 1.0])
    p.add_argument('--val-crops', type=int, choices=[0, 5, 10], default=0)
    p.add_argument('--crop-resize', type=int, default=256)
    p.add_argument('--train-cache', type=Path)
    p.add_argument('--val-cache', type=Path)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--cache', type=Path)
    p.add_argument('--optimizer', choices=['sgd', 'adam', 'capi_adamw'], default='sgd')
    p.add_argument('--warmup-updates', type=int, default=0)
    p.add_argument('--head-init', choices=['ibot', 'capi'], default='ibot')
    p.add_argument('--metric-diagnostics', action='store_true')
    p.add_argument('--report-crop-aggregates', action='store_true')
    p.add_argument('--standardize', action='store_true')
    p.add_argument('--balanced', action='store_true')
    p.add_argument('--loss', choices=['bce', 'asymmetric'], default='bce')
    p.add_argument('--regularization-sweep', action='store_true')
    p.add_argument('--manifest', type=Path, default=Path('output/evaluation/crisp-multilabel-v5-20261007_235427/manifests/pascal_voc.json'))
    args = p.parse_args()
    if args.resolution % 16 or args.resolution < 16:
        p.error('Resolution must be a positive multiple of patch size 16')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda'
    manifest = args.manifest
    root = '/mnt/fast/nobackup/scratch4weeks/jg02228/datasets'
    samples, classes, data_metadata = read_multilabel_manifest(manifest, root, 'pascal_voc', 20)
    checkpoint = Path('checkpoints/ibot_vit_small.pth')
    model, metadata = load_backbone(checkpoint, 'teacher', 'vit_small')
    # Match the production CPU head initialization after backbone construction.
    head_rng = torch.get_rng_state()
    model.to(device).eval()
    resolution = args.resolution
    if args.transform == 'square':
        transform = [T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC)]
    elif args.transform == 'center':
        transform = [T.Resize(round(resolution * 256 / 224), interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(resolution)]
    else:
        transform = [AspectPreservingSquare(resolution)]
    transform = T.Compose(transform + [T.ToTensor(), IMAGENET_NORMALIZE])
    cache_path = args.cache or args.output / 'features.pth'
    signature = {'model': metadata, 'manifest': data_metadata['sha256'], 'transform': args.transform, 'feature': 'float32 final-LayerNorm CLS last four + mean final patch tokens'}
    if resolution != 224 or args.transform == 'pad':
        signature['input_resolution'] = resolution
        signature['padding'] = 'RGB 124,116,104' if args.transform == 'pad' else None
    if args.train_crops or args.val_crops:
        signature['train_crops'] = args.train_crops
        signature['crop_scale'] = args.crop_scale
        signature['val_crops'] = args.val_crops
        signature['crop_resize'] = args.crop_resize
        signature['augmentation_note'] = 'RRC diagnostic uses a finite cache of augmented training views; not 500 independently redrawn epochs. Validation crop features are averaged, equivalent to averaging linear logits.'
    if args.train_cache or args.val_cache:
        signature['reused_split_caches'] = {k: str(v.resolve()) for k, v in [('train', args.train_cache), ('val', args.val_cache)] if v}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        assert cached['signature'] == signature
    else:
        cached = {'signature': signature}
        with torch.inference_mode():
            for split in ['train', 'val']:
                reuse = args.train_cache if split == 'train' else args.val_cache
                if reuse:
                    source = torch.load(reuse, map_location='cpu', weights_only=False)
                    assert source['signature']['model'] == metadata
                    assert source['signature']['manifest'] == data_metadata['sha256']
                    cached[split] = source[split]
                    continue
                feature_batch = max(8, min(96, int(96 * (224 / resolution) ** 4)))
                split_transform = transform
                view_count = 1
                if split == 'train' and args.train_crops:
                    split_transform = CropViews(resolution, train_views=args.train_crops, scale=tuple(args.crop_scale))
                    view_count = args.train_crops
                elif split == 'val' and args.val_crops:
                    split_transform = CropViews(resolution, val_views=args.val_crops, resize=args.crop_resize)
                    view_count = args.val_crops
                loader = DataLoader(MultilabelDataset(samples[split], classes, split_transform), batch_size=max(1, feature_batch // view_count), num_workers=4, pin_memory=True)
                features, flips, targets, views = [], [], [], []
                for i, (images, labels) in enumerate(loader):
                    images = images.to(device)
                    flat_images = images.flatten(0, 1) if images.ndim == 5 else images
                    vectors = []
                    for chunk in flat_images.split(feature_batch):
                        layers = model.get_intermediate_layers(chunk, n=4)
                        vectors.append(torch.cat([*[x[:, 0] for x in layers], layers[-1][:, 1:].mean(1)], dim=1).cpu())
                    vectors = torch.cat(vectors)
                    if view_count > 1:
                        vectors = vectors.reshape(len(labels), view_count, -1)
                        if split == 'train':
                            views.append(vectors)
                            features.append(vectors[:, 0])
                        else:
                            features.append(vectors.mean(1))
                            views.append(vectors)
                    else:
                        features.append(vectors)
                    if split == 'train' and not args.train_crops:
                        layers = model.get_intermediate_layers(images.flip(-1), n=4)
                        flips.append(torch.cat([*[x[:, 0] for x in layers], layers[-1][:, 1:].mean(1)], dim=1).cpu())
                    targets.append(labels)
                    if i % 10 == 0:
                        print(f'cache {split}: {i + 1}/{len(loader)}', flush=True)
                cached[split] = {'features': torch.cat(features), 'targets': torch.cat(targets)}
                if flips:
                    cached[split]['flipped'] = torch.cat(flips)
                if views:
                    cached[split]['views'] = torch.cat(views)
        torch.save(cached, cache_path)
    del model
    train_x = cached['train']['features'].to(device)
    flip_x = cached['train'].get('flipped', cached['train']['features']).to(device)
    augmented_x = cached['train']['views'].to(device) if 'views' in cached['train'] else None
    train_y = cached['train']['targets'].to(device)
    val_x = cached['val']['features'].to(device)
    val_y = cached['val']['targets'].numpy()
    val_views = cached['val']['views'].to(device) if args.report_crop_aggregates and 'views' in cached['val'] else None
    if args.report_crop_aggregates and val_views is None:
        raise ValueError('Crop aggregation requires individual cached validation views')
    lrs = [.001, .004, .01, .02, .04, .1, .2, .5] if args.optimizer == 'sgd' else [.0001, .0004, .001, .004, .01]
    decays = [0]
    if args.regularization_sweep:
        if args.optimizer == 'capi_adamw':
            lrs, decays = [.0001, .001, .004, .01], [0, .001, .01, .1]
        else:
            assert args.optimizer == 'sgd'
            lrs, decays = [.004, .01, .04, .1], [0, .0001, .001, .01, .1]
    if args.learning_rates:
        lrs = args.learning_rates
    modes = args.pooling or ['cls_mean_patch', 'cls_last', 'cls_last4', 'cls_last4_mean_patch']
    optimizer_name = {'sgd': 'SGD momentum 0.9', 'adam': 'Adam beta1=0.9 beta2=0.999 epsilon=1e-8', 'capi_adamw': 'AdamW beta1=0.9 beta2=0.95 epsilon=1e-8; weight-only decoupled decay'}[args.optimizer]
    loss_name = 'ASL gamma_negative=4 gamma_positive=0 clip=0.05, detached focal weights, mean known-label reduction' if args.loss == 'asymmetric' else ('train-frequency balanced mean known-label BCE' if args.balanced else 'mean known-label BCE')
    variants = [{'pooling': mode, 'lr': lr, 'weight_decay': decay, 'warmup_updates': args.warmup_updates, 'optimizer': optimizer_name, 'loss': loss_name, 'normalization': 'final LayerNorm + training-set StandardScaler' if args.standardize else 'final LayerNorm'} for mode in modes for lr in lrs for decay in decays]
    h, width = len(variants), train_x.shape[1]
    masks = torch.zeros(h, 1, width, device=device)
    for i, variant in enumerate(variants):
        mode = variant['pooling']
        if mode == 'cls_mean_patch': masks[i, :, 1152:] = 1
        elif mode == 'cls_last': masks[i, :, 1152:1536] = 1
        elif mode == 'cls_last4': masks[i, :, :1536] = 1
        else: masks[i] = 1
    lr_tensor = torch.tensor([v['lr'] for v in variants], device=device)[:, None, None]
    decay_tensor = torch.tensor([v['weight_decay'] for v in variants], device=device)[:, None, None]
    result = {'status': 'running', 'protocol': signature, 'dataset': data_metadata, 'epochs': args.epochs, 'batch_size': 1024, 'seed': 0, 'initialization': 'Production CPU linear head after teacher backbone construction; identical initialization across learning rates within pooling', 'caveat': 'Exploratory validation/test sweep; do not infer CRISP equivalence from matching its score. Frozen-feature forward/backward, no backbone training, no projection head.', 'results': []}
    if args.head_init == 'capi':
        result['initialization'] = 'CAPI nn.init.trunc_normal_(std=0.02), zero bias; same CPU seed after teacher construction'
    start = time.monotonic()
    for regime in ['1shot', 'full']:
        indices = sample_few_shot_indices(cached['train']['targets'].numpy(), 1, 0)[0] if regime == '1shot' else list(range(len(train_x)))
        x, xf, y = train_x[indices], flip_x[indices], train_y[indices]
        xa = augmented_x[indices] if augmented_x is not None else None
        validation_features = val_x
        validation_views = val_views
        if args.standardize:
            if xa is not None:
                raise ValueError('StandardScaler with cached RRC views is not supported by this diagnostic')
            # Only the actual training subset sets the scaler, including flips.
            mu = (x.mean(0) + xf.mean(0)) / 2
            variance = ((x - mu).square().mean(0) + (xf - mu).square().mean(0)) / 2
            std = variance.sqrt().clamp_min(1e-6)
            x, xf = (x - mu) / std, (xf - mu) / std
            validation_features = (val_x - mu) / std
            if val_views is not None:
                validation_views = (val_views - mu) / std
        known, target = y >= 0, y.clamp_min(0)
        weights = known.to(torch.float32)
        if args.balanced:
            positive, negative = ((y == 1).sum(0).float(), (y == 0).sum(0).float())
            total = positive + negative
            weights *= torch.where(target == 1, total / (2 * positive.clamp_min(1)), total / (2 * negative.clamp_min(1)))
        w = torch.zeros(h, 20, width, device=device)
        b = torch.zeros(h, 20, device=device)
        for i in range(h):
            torch.set_rng_state(head_rng)
            active = masks[i, 0].bool()
            initial = torch.nn.Linear(int(active.sum()), 20)
            if args.head_init == 'capi':
                torch.nn.init.trunc_normal_(initial.weight, std=.02)
            else:
                torch.nn.init.normal_(initial.weight, std=.01)
            torch.nn.init.zeros_(initial.bias)
            w[i, :, active] = initial.weight.detach().to(device)
        vw, vb = torch.zeros_like(w), torch.zeros_like(b)
        sw, sb = torch.zeros_like(w), torch.zeros_like(b)
        updates = 0
        for epoch in range(args.epochs):
            torch.manual_seed(epoch)
            order = torch.randperm(len(x), device=device)
            scale = .5 * (1 + math.cos(math.pi * epoch / args.epochs))
            for batch in order.split(1024):
                if args.warmup_updates:
                    total_updates = args.epochs * math.ceil(len(x) / 1024)
                    warmup = min(args.warmup_updates, max(1, total_updates // 2))
                    scale = updates / max(1, warmup - 1) if updates < warmup else .5 * (1 + math.cos(math.pi * (updates - warmup) / max(1, total_updates - warmup - 1)))
                if xa is None:
                    xx = torch.where((torch.rand(len(batch), 1, device=device) < .5), xf[batch], x[batch])
                else:
                    xx = xa[batch, torch.randint(xa.shape[1], (len(batch),), device=device)]
                yy, kk = target[batch], known[batch]
                logits = F.linear(xx, w.flatten(0, 1), b.flatten()).reshape(len(batch), h, 20)
                if args.loss == 'asymmetric':
                    with torch.enable_grad():
                        logits.requires_grad_(True)
                        positive = logits.sigmoid()
                        negative = (1 - positive + .05).clamp(max=1)
                        truth = yy[:, None]
                        focal = (1 - (positive * truth + negative * (1 - truth))).pow(4 * (1 - truth)).detach()
                        losses = -(truth * positive.clamp_min(1e-8).log() + (1 - truth) * negative.clamp_min(1e-8).log()) * focal
                        loss = (losses * weights[batch, None]).sum() / kk.sum()
                        residual = torch.autograd.grad(loss, logits)[0]
                    logits = logits.detach()
                else:
                    residual = (logits.sigmoid() - yy[:, None]) * weights[batch, None] / kk.sum()
                gw = residual.flatten(1).T.matmul(xx).reshape(h, 20, width) * masks
                gb = residual.sum(0)
                if args.optimizer != 'capi_adamw':
                    gw = gw + decay_tensor * w
                    gb = gb + decay_tensor.squeeze(-1) * b
                updates += 1
                if args.optimizer == 'sgd':
                    vw.mul_(.9).add_(gw)
                    vb.mul_(.9).add_(gb)
                    w.add_(-scale * lr_tensor * vw)
                    b.add_(-scale * lr_tensor.squeeze(-1) * vb)
                else:
                    beta2 = .95 if args.optimizer == 'capi_adamw' else .999
                    vw.mul_(.9).add_(gw, alpha=.1)
                    vb.mul_(.9).add_(gb, alpha=.1)
                    sw.mul_(beta2).addcmul_(gw, gw, value=1 - beta2)
                    sb.mul_(beta2).addcmul_(gb, gb, value=1 - beta2)
                    if args.optimizer == 'capi_adamw':
                        w.mul_(1 - scale * lr_tensor * decay_tensor)
                    w.add_(-scale * lr_tensor * (vw / (1 - .9 ** updates)) / ((sw / (1 - beta2 ** updates)).sqrt() + 1e-8))
                    b.add_(-scale * lr_tensor.squeeze(-1) * (vb / (1 - .9 ** updates)) / ((sb / (1 - beta2 ** updates)).sqrt() + 1e-8))
            if (epoch + 1) % 100 == 0:
                print(f'{regime}: epoch {epoch + 1}/{args.epochs}, elapsed {time.monotonic() - start:.1f}s', flush=True)
        predictions = F.linear(validation_features, w.flatten(0, 1), b.flatten()).reshape(len(val_x), h, 20).cpu().numpy()
        if validation_views is not None:
            crop_predictions = F.linear(validation_views, w.flatten(0, 1), b.flatten()).reshape(len(val_x), validation_views.shape[1], h, 20)
            maximum_predictions = crop_predictions.max(1).values.cpu().numpy()
            mean_probability_predictions = crop_predictions.sigmoid().mean(1).cpu().numpy()
        for i, variant in enumerate(variants):
            metrics = multilabel_metrics(val_y, predictions[:, i], classes, 'pascal_voc')
            row = {'regime': regime, **variant, 'map_percent': metrics['map_percent']}
            if validation_views is not None:
                row['max_crop_map_percent'] = multilabel_metrics(val_y, maximum_predictions[:, i], classes, 'pascal_voc')['map_percent']
                row['mean_crop_probability_map_percent'] = multilabel_metrics(val_y, mean_probability_predictions[:, i], classes, 'pascal_voc')['map_percent']
                row['crop_aggregation_note'] = 'All three are standard per-class VOC mAP. map_percent uses mean logits; other fields use classwise max crop logits or mean sigmoid probabilities. Same trained linear head; extra test-time views.'
            if args.metric_diagnostics:
                from sklearn.metrics import average_precision_score
                scores = predictions[:, i]
                known_validation = val_y >= 0
                class_positives = (val_y == 1).sum(0)
                class_aps = np.array([metrics['average_precision_by_class'][name] for name in classes])
                row['micro_ap_percent'] = float(100 * average_precision_score(val_y[known_validation], scores[known_validation]))
                row['frequency_weighted_voc_ap_percent'] = float(100 * np.average(class_aps, weights=class_positives))
                row['macro_noninterpolated_ap_percent'] = float(100 * np.mean([average_precision_score(val_y[known_validation[:, c], c], scores[known_validation[:, c], c]) for c in range(len(classes))]))
                order = np.argsort(-np.where(known_validation, scores, -np.inf), axis=1, kind='stable')
                ordered = np.take_along_axis(val_y == 1, order, axis=1)
                cumulative = np.cumsum(ordered, axis=1) / np.arange(1, len(classes) + 1)[None]
                count = ordered.sum(1)
                per_image = np.divide((cumulative * ordered).sum(1), count, out=np.zeros(len(count)), where=count > 0)
                row['samples_ap_percent'] = float(100 * per_image.mean())
                row['metric_note'] = 'Only map_percent is standard per-class macro VOC mAP; micro, frequency-weighted and per-image AP are different statistics.'
            if 'VOC2007' in data_metadata['annotation_source']:
                row['voc2007_11point_map_percent'] = voc2007_map(val_y, predictions[:, i])
            result['results'].append(row)
            print(json.dumps(row), flush=True)
        save(args.output / 'results.json', result)
        if args.metric_diagnostics:
            torch.save({'targets': torch.from_numpy(val_y), 'logits': torch.from_numpy(predictions), 'variants': variants, 'regime': regime}, args.output / f'{regime}_predictions.pth')
    result['status'] = 'completed'
    result['elapsed_seconds'] = time.monotonic() - start
    save(args.output / 'results.json', result)


if __name__ == '__main__':
    main()
