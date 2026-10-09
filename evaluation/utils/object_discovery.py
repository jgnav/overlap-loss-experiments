"""Image and box conventions for DINOv3's published TokenCut protocol."""

PUBLISHED_THRESHOLDS = tuple(i / 20 for i in range(9))
# Extended uniformly for every model; retain the published range in the reports.
THRESHOLDS = tuple(i / 20 for i in range(20))


def native_image_tensor(image, patch_size=16):
    """Official TokenCut preprocessing: normalize, then zero-pad; no resize."""
    import torch
    from torchvision.transforms.functional import normalize, to_tensor

    image = image.convert("RGB")
    tensor = normalize(to_tensor(image), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    channels, height, width = tensor.shape
    padded = torch.zeros(channels, (height + patch_size - 1) // patch_size * patch_size,
                         (width + patch_size - 1) // patch_size * patch_size)
    padded[:, :height, :width] = tensor
    return padded, (height, width)


def box_iou(prediction, boxes):
    """Continuous xyxy IoU without inclusive-coordinate +1 adjustments."""
    import numpy as np

    pred = np.asarray(prediction, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(pred[2:], boxes[:, 2:]) -
                              np.maximum(pred[:2], boxes[:, :2])).prod(axis=1)
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    pred_area = np.maximum(0, pred[2:] - pred[:2]).prod()
    return intersection / np.maximum(pred_area + areas - intersection, 1e-12)


def voc_boxes(annotation):
    """Retain difficult/truncated objects; convert VOC's 1-based minima."""
    return [[int(obj.findtext("bndbox/xmin")) - 1,
             int(obj.findtext("bndbox/ymin")) - 1,
             int(obj.findtext("bndbox/xmax")), int(obj.findtext("bndbox/ymax"))]
            for obj in annotation.findall("object")]


def coco_boxes(annotations):
    """Official TokenCut: exclude crowd and round converted xyxy coordinates."""
    boxes = []
    for annotation in annotations:
        if annotation["iscrowd"]:
            continue
        x, y, width, height = annotation["bbox"]
        boxes.append([int(round(value)) for value in (x, y, x + width, y + height)])
    return boxes


def predict_boxes(patch_tokens, dims, original_size, thresholds=THRESHOLDS):
    """Run unchanged official NCut; its dummy CLS input is always discarded."""
    import torch
    from evaluation.vendor.tokencut.object_discovery import ncut

    if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
        raise ValueError("Expected one image's patch outputs [1, patches, channels]")
    if patch_tokens.shape[1] != dims[0] * dims[1]:
        raise ValueError("Patch grid does not match patch outputs")
    if patch_tokens.shape[1] < 3:
        raise ValueError("Official TokenCut's eigenvector solve requires at least three patches")
    feats = torch.cat((torch.zeros_like(patch_tokens[:, :1]), patch_tokens.float()), dim=1)
    height, width = original_size
    predictions = {}
    for threshold in thresholds:
        box, _, _, seed, _, _ = ncut(feats, dims, [16, 16], (3, height, width), tau=threshold, eps=1e-5)
        predictions[f"{threshold:.2f}"] = {"bbox_xyxy": box.tolist(), "seed_patch": int(seed)}
    return predictions
