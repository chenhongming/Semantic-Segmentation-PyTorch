import os
import torch
import platform
from thop import profile, clever_format

from config.config import cfg
from utils.utils import setup_logger
logger = setup_logger('misc-logger')


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def params_flops(net, size, device):
    if isinstance(size, int):
        _size = [size, size]
    elif isinstance(size, list) or isinstance(size, tuple):
        _size = size
    else:
        raise RuntimeError("size error.")
    img = torch.randn(1, 3, _size[0], _size[1]).to(device)
    flops, params = profile(net.to(device), inputs=(img, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    logger.info("Model: {}({}) | Flops: {} | Params: {}".format(net.__class__.__name__, cfg.MODEL.BACKBONE_NAME,
                                                                flops, params) + '\n')
    del net


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def device_info(device):
    # device = 'cpu' or 'cuda'
    info = 'torch: {} ||'.format(torch.__version__) + ' python: {} ||'.format(platform.python_version())
    if device == 'cuda':
        p = torch.cuda.get_device_properties(0)
        info += ' device: GPU ||'
        msg = '\t\t({}) [memory: {}'.format(p.name, p.total_memory / 1024 ** 2) + ' MB]'
    else:
        info += ' device: CPU ||'
    logger.info('*' * 48)
    logger.info(info)
    if device == 'cuda':
        logger.info(msg)
    logger.info('*' * 48)
