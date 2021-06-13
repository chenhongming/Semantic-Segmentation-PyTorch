import os
import torch
import logging
from collections import OrderedDict
from torch.hub import load_state_dict_from_url as load_url  # noqa: F401

from utils.registry import Registry
from utils.utils import setup_logger, root_path
from config.config import cfg

BACKBONE_REGISTRY = Registry('backbone')

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',

    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',

    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',

    'shufflenet_v2_x0_5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenet_v2_x1_0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}


def set_backbone():
    backbone_name = cfg.MODEL.BACKBONE_NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)()
    if cfg.MODEL.BACKBONE_PRETRAINED:
        backbone = load_pretrain_backbone(backbone, backbone_name)
    return backbone


def load_pretrain_backbone(backbone, backbone_name):
    pretrained_file = root_path() + cfg.MODEL.BACKBONE_WEIGHT

    logger = setup_logger('build-logger')
    if os.path.isfile(pretrained_file) and os.path.splitext(pretrained_file)[-1] == '.pth':
        logger.info("Load backbone pretrained model from {}".format(cfg.MODEL.BACKBONE_WEIGHT))
        ckpt = torch.load(pretrained_file)
        backbone_dict = backbone.state_dict()
        matched_weights, unmatched_weights = weight_filler(ckpt, backbone_dict)
        logger.info("Unmatched backbone layers: {}".format(unmatched_weights))
        backbone.load_state_dict(matched_weights, strict=False)
        logger.info('Loaded!')
    elif backbone_name in model_urls:
        # default download path using torch.hub import load_state_dict_from_url method
        cached_file = os.path.expanduser("~/.cache/torch/hub/checkpoints/" + model_urls[backbone_name].split('/')[-1])
        if os.path.isfile(cached_file) and os.path.splitext(cached_file)[-1] == '.pth':
            logger.info("Load backbone pretrained model from {}".format(cached_file))
            ckpt = torch.load(cached_file)
            backbone_dict = backbone.state_dict()
            matched_weights, unmatched_weights = weight_filler(ckpt, backbone_dict)
            logger.info("Unmatched backbone layers: {}".format(unmatched_weights))
            backbone.load_state_dict(matched_weights, strict=False)
            logger.info('Loaded!')
        else:
            logger.info("Load backbone pretrained model from url {}".format(model_urls[backbone_name]))
            try:
                ckpt = load_url(model_urls[backbone_name])
                backbone_dict = backbone.state_dict()
                matched_weights, unmatched_weights = weight_filler(ckpt, backbone_dict)
                logger.info("Unmatched backbone layers: {}".format(unmatched_weights))
                backbone.load_state_dict(matched_weights, strict=False)
                logger.info('Loaded!')
            except Exception as e:
                logger.info(e)
                logger.info("Use torch download pretrained model failed!")
    else:
        logger.info("{} has no pretrained model and use kaiming_normal init...".format(backbone_name))
    return backbone


def weight_filler(ckpt_dict, model_dict):
    matched_weights = OrderedDict()
    unmatched_weights = []
    for k, v in ckpt_dict.items():
        if k in model_dict.keys():
            if v.shape == model_dict[k].shape:
                matched_weights[k] = v
            else:
                unmatched_weights.append([k, str(v.shape), str(model_dict[k].shape)])
        else:
            unmatched_weights.append(k)
    return matched_weights, unmatched_weights
