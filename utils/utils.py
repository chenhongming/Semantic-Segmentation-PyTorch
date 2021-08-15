import os
import torch
import random
import logging

import numpy as np


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
    cur_path = os.path.dirname(__file__)
    return cur_path[:cur_path.find('Semantic Segmentation PyTorch') + len('Semantic Segmentation PyTorch')+1]


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
