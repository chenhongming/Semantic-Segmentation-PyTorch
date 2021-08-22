import os
import torch
from thop import profile, clever_format

from config.config import cfg
from utils.utils import setup_logger


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def params_flops(net, size, device):
    logger = setup_logger('misc-logger')
    if isinstance(size, int):
        _size = [size, size]
    elif isinstance(size, list) or isinstance(size, tuple):
        _size = size
    else:
        raise RuntimeError("size error.")
    img = torch.randn(1, 3, _size[0], _size[1]).to(device)
    flops, params = profile(net, inputs=(img, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    logger.info("Model: {}({}) | Flops: {} | Params: {}".format(net.__class__.__name__, cfg.MODEL.BACKBONE_NAME,
                                                                flops, params) + '\n')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']