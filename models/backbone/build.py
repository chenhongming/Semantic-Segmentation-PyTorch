import re
import os
import torch
from collections import OrderedDict
from torch.hub import load_state_dict_from_url as load_url  # noqa: F401

from config.config import cfg
from utils.registry import Registry
from utils.utils import setup_logger, root_path
from utils.distributed import is_main_process

BACKBONE_REGISTRY = Registry('backbone')
logger = setup_logger('build-logger')

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

    'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',

    'shufflenet_v2_x0_5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenet_v2_x1_0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',

    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',

    "mnasnet0_5": "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
    "mnasnet1_0": "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
}


def set_backbone():
    backbone_name = cfg.MODEL.BACKBONE_NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)()
    if cfg.MODEL.BACKBONE_PRETRAINED and is_main_process():
        backbone = load_pretrain_backbone(backbone, backbone_name)
    return backbone


def load_pretrain_backbone(backbone, backbone_name):
    pretrained_file = root_path() + cfg.MODEL.BACKBONE_WEIGHT
    if os.path.isfile(pretrained_file) and os.path.splitext(pretrained_file)[-1] == '.pth':
        logger.info("Load backbone pretrained model from {}".format(cfg.MODEL.BACKBONE_WEIGHT))
        ckpt = torch.load(pretrained_file)
        if backbone_name.startswith('densenet'):
            ckpt = _load_densenet_dict(ckpt)
        backbone_dict = backbone.state_dict()
        matched_weights, unmatched_weights = weight_filler(ckpt, backbone_dict)
        logger.info("Unmatched backbone layers: {}".format(unmatched_weights))
        logger.info('Loaded!')
        backbone.load_state_dict(matched_weights, strict=False)
    elif backbone_name in model_urls:
        # default download path using torch.hub import load_state_dict_from_url method
        cached_file = os.path.expanduser("~/.cache/torch/hub/checkpoints/" + model_urls[backbone_name].split('/')[-1])
        if os.path.isfile(cached_file) and os.path.splitext(cached_file)[-1] == '.pth':
            logger.info("Load backbone pretrained model from {}".format(cached_file))
            ckpt = torch.load(cached_file)
            if backbone_name.startswith('densenet'):
                ckpt = _load_densenet_dict(ckpt)
            backbone_dict = backbone.state_dict()
            matched_weights, unmatched_weights = weight_filler(ckpt, backbone_dict)
            logger.info("Unmatched backbone layers: {}".format(unmatched_weights))
            logger.info('Loaded!')
            backbone.load_state_dict(matched_weights, strict=False)
        else:
            logger.info("Load backbone pretrained model from url {}".format(model_urls[backbone_name]))
            try:
                ckpt = load_url(model_urls[backbone_name])
                if backbone_name.startswith('densenet'):
                    ckpt = _load_densenet_dict(ckpt)
                backbone_dict = backbone.state_dict()
                matched_weights, unmatched_weights = weight_filler(ckpt, backbone_dict)
                logger.info("Unmatched backbone layers: {}".format(unmatched_weights))
                logger.info('Loaded!')
                backbone.load_state_dict(matched_weights, strict=False)
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


# modified from torchvision.models.densenet.py
def _load_densenet_dict(state_dict):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def load_trained_model(model):
    model_file = root_path() + cfg.MODEL.MODEL_WEIGHT
    suffix = os.path.splitext(model_file)[-1]
    if os.path.isfile(model_file) and suffix == '.pth':
        ckpt = torch.load(model_file)['state_dict']
        model_dict = model.state_dict()
        matched_weights, unmatched_weights = weight_filler(ckpt, model_dict)
        logger.info("Unmatched model layers: {}".format(unmatched_weights))
        model.load_state_dict(matched_weights, strict=True)
        logger.info('Loaded trained model weights!')
    else:
        raise Exception("{} is not a valid pth file".format(model_file))
    return model
