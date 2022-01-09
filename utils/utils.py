import os
import torch
import random
import logging

import numpy as np
from PIL import ImageOps


def setup_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def root_path():
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'


def set_norm(norm):
    if norm == 'bn':
        return torch.nn.BatchNorm2d
    elif norm == 'syncbn':
        return torch.nn.SyncBatchNorm
    else:
        raise AttributeError


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# fix batch size > 1 for training using torch.utils.data.DataLoader
def unified_size(image, mask, crop_size, padding=(0, 0, 0), ignore_label=255):
    w, h = image.size
    if isinstance(crop_size, int):
        _size = [crop_size, crop_size]
    elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
        _size = crop_size
    else:
        raise RuntimeError("size error.")
    crop_w, crop_h = _size
    pad_w = max((crop_w - w), 0)
    pad_h = max((crop_h - h), 0)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w // 2, pad_h // 2, (crop_w - w) - pad_w // 2, (crop_h - h) - pad_h // 2)
        image = ImageOps.expand(image, border=border, fill=padding)
        mask = ImageOps.expand(mask, border=border, fill=ignore_label)
    else:
        w_off = (w - crop_w) // 2
        h_off = (h - crop_h) // 2
        border = (w_off, h_off, (w - crop_w) - w_off, (h - crop_h) - h_off)
        image = ImageOps.crop(image, border=border)
        mask = ImageOps.crop(mask, border=border)
    return image, mask
