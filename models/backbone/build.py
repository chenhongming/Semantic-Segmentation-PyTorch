import os
import torch
from torch.hub import load_state_dict_from_url as load_url  # noqa: F401

from utils.registry import Registry
from utils.utils import setup_logger
from config.config import cfg

BACKBONE_REGISTRY = Registry('backbone')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def set_backbone():
    backbone_name = cfg.MODEL.BACKBONE_NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)()
    if cfg.MODEL.BACKBONE_PRETRAINED:
        backbone = load_pretrain_backbone(backbone, backbone_name)
    return backbone


def load_pretrain_backbone(backbone, backbone_name):
    pretrained_file = root_path() + cfg.MODEL.BACKBONE_WEIGHT
    logger = setup_logger()
    if os.path.isfile(pretrained_file):
        logger.info("Load backbone pretrained model from {}".format(cfg.MODEL.BACKBONE_WEIGHT))
        return backbone.load_state_dict(torch.load(pretrained_file))
    elif backbone_name in model_urls:
        logger.info("Load backbone pretrained model from url {}".format(model_urls[backbone_name]))
        try:
            return backbone.load_state_dict(load_url(model_urls[backbone_name]), strict=False)
        except Exception as e:
            logger.info(e)
            logger.info("Use torch download pretrained model failed!")
        logger.info("{} has no pretrained model and use kaiming_normal init...".format(backbone_name))
        return backbone


def root_path():
    cur_path = os.path.dirname(__file__)
    return cur_path[:cur_path.find('Semantic Segmentation PyTorch') + len('Semantic Segmentation PyTorch')+1]


if __name__ == '__main__':
    pass
