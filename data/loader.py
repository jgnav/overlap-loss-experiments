import random
import math
import numpy as np

from torchvision.datasets import ImageFolder


class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index


class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var,
                 pred_aspect_ratio, pred_shape='block', pred_start_epoch=0,
                 **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0
        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(
                self.pred_ratio - self.pred_ratio_var,
                self.pred_ratio + self.pred_ratio_var,
            ) if self.pred_ratio_var > 0 else self.pred_ratio
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        (images, global_crop_boxes), target = super(
            ImageFolderMask, self
        ).__getitem__(index)

        masks = []
        for view, img in enumerate(images):
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except Exception:
                continue

            if (getattr(self.transform, 'region_views', 'global') == 'global_unmasked'
                    and view >= len(images) - 4):
                # These crops have no masked-patch objective. Avoid generating
                # unused random masks, and retain explicit unmasked metadata.
                masks.append(np.zeros((H, W), dtype=bool))
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count
                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)
                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1
                        if delta > 0:
                            break
                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
            else:
                assert False

            masks.append(mask)

        return images, target, masks, global_crop_boxes
