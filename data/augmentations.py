import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from utils import training as utils


class DataAugmentationiBOT:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
        global_crop_size,
        local_crop_size,
        region_views="global",
        record_local_geometry=False,
    ):
        color_jitter = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

        self.global_crops_number = global_crops_number
        self.global_crops_scale = global_crops_scale
        self.global_crop_size = global_crop_size
        self.global_transfo1 = transforms.Compose(
            [
                color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ]
        )
        self.global_transfo2 = transforms.Compose(
            [
                color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ]
        )
        self.local_crops_number = local_crops_number
        self.record_local_geometry = record_local_geometry or region_views in ("global_local", "local")
        self.region_views = region_views
        self.local_crops_scale = local_crops_scale
        self.local_crop_size = local_crop_size
        self.local_transfo = transforms.Compose(
            [
                color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    @staticmethod
    def _spatial_transform(image, size, scale):
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image,
            scale=scale,
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
        image_width, image_height = image.size
        crop = F.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [size, size],
            interpolation=Image.BICUBIC,
            antialias=True,
        )
        flipped = torch.rand(1).item() < 0.5
        if flipped:
            crop = F.hflip(crop)

        crop_box = torch.tensor(
            [
                left / image_width,
                top / image_height,
                (left + width) / image_width,
                (top + height) / image_height,
                float(flipped),
            ],
            dtype=torch.float32,
        )
        return crop, crop_box

    def _global_crop(self, image, transform):
        crop, crop_box = self._spatial_transform(
            image,
            self.global_crop_size,
            self.global_crops_scale,
        )
        return transform(crop), crop_box

    def _local_crop(self, image):
        crop, crop_box = self._spatial_transform(
            image,
            self.local_crop_size,
            self.local_crops_scale,
        )
        return self.local_transfo(crop), crop_box

    def __call__(self, image):
        first_crop, first_box = self._global_crop(image, self.global_transfo1)
        crops = [first_crop]
        global_crop_boxes = [first_box]
        for _ in range(self.global_crops_number - 1):
            crop, crop_box = self._global_crop(image, self.global_transfo2)
            crops.append(crop)
            global_crop_boxes.append(crop_box)
        for _ in range(self.local_crops_number):
            crop, crop_box = self._local_crop(image)
            crops.append(crop)
            if self.record_local_geometry:
                global_crop_boxes.append(crop_box)
        if self.region_views == "global_unmasked":
            # Separate region-only views, after the original two globals and
            # all locals. They never contribute to DINO or ordinary iBOT.
            for view in range(4):
                crop, crop_box = self._global_crop(
                    image, self.global_transfo1 if view == 0 else self.global_transfo2
                )
                crops.append(crop)
                global_crop_boxes.append(crop_box)
        return crops, torch.stack(global_crop_boxes)
