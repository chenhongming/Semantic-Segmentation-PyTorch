import random
from PIL import Image
import torchvision.transforms.functional as F


class Compose:
    # Composes augmentations
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, mask):
        assert image.size == mask.size
        for t in self.augmentations:
            image, mask = t(image, mask)
        return image, mask


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        assert img.size == mask.size
        assert 0 <= self.p <= 1
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask
        return img, mask


class RandomResize:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img, mask):
        assert img.size == mask.size and len(self.ratio) == 2
        w, h = img.size
        ow = int(w * random.uniform(self.ratio[0], self.ratio[1]))
        oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        return img, mask


class RandomCrop:
    def __init__(self, crop_size=(640, 512)):
        if isinstance(crop_size, int):
            self.crop_h = crop_size
            self.crop_w = crop_size
        elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
            self.crop_h = crop_size[0]
            self.crop_w = crop_size[1]
        else:
            raise RuntimeError("Crop size error.")

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if w >= self.crop_w and h >= self.crop_h:
            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)
            img = img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            mask = mask.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
        return img, mask


# Randomly rotate image & mask with rotate factor in (rotate_min, rotate_max)
class RandomRotate:
    def __init__(self, rotate=(-10, 10), padding=(0, 0, 0), ignore_label=255):
        self.rotate = rotate
        self.padding = padding
        self.ignore_label = ignore_label

    def __call__(self, img, mask):
        assert img.size == mask.size
        assert len(self.rotate) == 2 and len(self.padding) == 3
        rotate_degree = random.randint(self.rotate[0], self.rotate[1])
        img = F.affine(img, angle=rotate_degree, interpolation=F.InterpolationMode.BILINEAR, fill=self.padding,
                       translate=(0, 0), scale=1.0, shear=0.0)
        mask = F.affine(mask, angle=rotate_degree, interpolation=F.InterpolationMode.NEAREST, fill=self.ignore_label,
                        translate=(0, 0), scale=1.0, shear=0.0)
        return img, mask
