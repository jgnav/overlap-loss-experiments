"""Mask-conditioned VOC2012 semantic region retrieval (custom analysis).

Train images are the gallery; disjoint validation images are queries. Semantic
class masks pool frozen final-LayerNorm patch features, then L2-normalize region
vectors. Neither support labels nor validation masks fit a model or transform.
"""
import argparse
import json
import hashlib
from pathlib import Path


def retrieval_metrics(query, query_labels, gallery, gallery_labels):
    import numpy as np
    query_labels, gallery_labels = np.asarray(query_labels), np.asarray(gallery_labels)
    scores = query @ gallery.T
    rows = []
    for i, label in enumerate(query_labels):
        order = np.argsort(-scores[i], kind='stable')
        hits = gallery_labels[order] == label
        npositives = int(hits.sum())
        if not npositives:
            raise ValueError(f'Class {label} has no positive gallery regions')
        ap = float((np.cumsum(hits) / np.arange(1, len(hits) + 1))[hits].sum() / npositives)
        rows.append({'class': int(label), 'AP': ap,
                     **{f'Recall@{k}': float(hits[:k].any()) for k in (1, 5, 10)}})
    keys = ['AP', 'Recall@1', 'Recall@5', 'Recall@10']
    per_class = {str(c): {k: float(np.mean([r[k] for r in rows if r['class'] == c])) * 100
                         for k in keys} for c in sorted(set(query_labels))}
    return {'micro_percent': {k: float(np.mean([r[k] for r in rows])) * 100 for k in keys},
            'macro_percent': {k: float(np.mean([r[k] for r in per_class.values()])) for k in keys},
            'per_class_percent': per_class, 'queries': len(query_labels), 'gallery_regions': len(gallery_labels)}


def extract(model, voc, ids, device, side):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms as T
    transform = T.Compose([T.Resize((side, side)), T.ToTensor(),
                           T.Normalize((.485, .456, .406), (.229, .224, .225))])
    vectors, labels, records = [], [], []
    model.eval()
    with torch.inference_mode():
        for j, name in enumerate(ids):
            with Image.open(voc / f'JPEGImages/{name}.jpg') as im:
                image = transform(im.convert('RGB')).unsqueeze(0).to(device)
            with Image.open(voc / f'SegmentationClass/{name}.png') as im:
                mask = np.array(im.resize((side, side), resample=Image.Resampling.NEAREST))
            features = model.get_intermediate_layers(image, n=1)[0][:, 1:]
            grid = side // 16
            assert features.shape == (1, grid * grid, 384)
            for c in sorted(set(np.unique(mask)) - {0, 255}):
                binary = torch.tensor(mask == c, dtype=torch.float32, device=device)[None, None]
                weights = F.avg_pool2d(binary, kernel_size=16, stride=16).flatten()
                if not weights.sum().item():
                    continue
                pooled = (features[0] * weights[:, None]).sum(0) / weights.sum()
                vector = F.normalize(pooled, dim=0)
                if not torch.isfinite(vector).all() or vector.norm() < .99:
                    raise RuntimeError('Invalid region descriptor')
                vectors.append(vector.cpu().numpy()); labels.append(int(c))
                records.append({'image': name, 'class': int(c), 'pixels': int((mask == c).sum())})
            if j % 100 == 0:
                print(f'Regions: {j}/{len(ids)} images, {len(labels)} regions', flush=True)
    return np.stack(vectors), np.array(labels), records


def main():
    import numpy as np
    import torch
    from evaluation.utils.common import load_backbone
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--voc', type=Path, required=True)
    p.add_argument('--image-size', type=int, default=448)
    a = p.parse_args()
    assert a.image_size % 16 == 0
    a.output.mkdir(parents=True, exist_ok=True)
    train = (a.voc / 'ImageSets/Segmentation/train.txt').read_text().split()
    val = (a.voc / 'ImageSets/Segmentation/val.txt').read_text().split()
    assert len(train) == 1464 and len(val) == 1449 and not set(train) & set(val)
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU is required')
    torch.set_num_threads(4)
    model, metadata = load_backbone(a.checkpoint)
    assert metadata['architecture'] == 'vit_small' and metadata['patch_size'] == 16
    model.cuda()
    gallery, gl, gallery_records = extract(model, a.voc, train, 'cuda', a.image_size)
    queries, ql, query_records = extract(model, a.voc, val, 'cuda', a.image_size)
    np.savez_compressed(a.output / 'descriptors.npz', gallery=gallery, gallery_labels=gl,
                        queries=queries, query_labels=ql)
    protocol = {'name': 'custom mask-conditioned semantic class-region retrieval',
                'input': [a.image_size, a.image_size], 'feature': 'teacher final LayerNorm patch tokens',
                'pooling': 'patch area-weighted semantic masks, then region L2 normalization',
                'gallery': 'VOC2012 train, 1464 images', 'queries': 'VOC2012 val, 1449 images',
                'background_and_void': 'excluded', 'training_or_test_time_augmentation': False,
                'split_sha256': hashlib.sha256(('\n'.join(train) + '\n' + '\n'.join(val)).encode()).hexdigest()}
    result = {'checkpoint': metadata, 'protocol': protocol,
              'metrics': retrieval_metrics(queries, ql, gallery, gl),
              'gallery_regions': gallery_records, 'query_regions': query_records, 'complete': True}
    tmp = a.output / 'results.json.tmp'
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(a.output / 'results.json')
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
