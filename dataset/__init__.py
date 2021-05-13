from .augmentations import *
from .dataset import JsonDataset


def set_augmentations(cfg):
    return Compose(
        [
            RandomFlip(cfg.TRAIN.PROB),
            RandomResize(cfg.TRAIN.RATIO),
            RandomCrop(cfg.TRAIN.CROP_SIZE),
            RandomRotate(cfg.TRAIN.ROTATE, cfg.TRAIN.PADDING, cfg.TRAIN.IGNORE_LABEL)
        ]
    )
