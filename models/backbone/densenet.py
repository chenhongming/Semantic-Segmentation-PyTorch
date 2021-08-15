import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .build import BACKBONE_REGISTRY
from config.config import cfg
from utils.utils import set_norm

__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

# Specification
densenet_cfg = {121: (64, 32, [6, 12, 24, 16]),
                161: (96, 48, [6, 12, 36, 24]),
                169: (64, 32, [6, 12, 32, 32]),
                201: (64, 32, [6, 12, 48, 32])}


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 padding=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1,
                                           padding=0, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,
                                           padding=padding, dilation=dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,
                 padding=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, padding, dilation, norm_layer)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, stride=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64):
        super().__init__()

        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.drop_rate = cfg.MODEL.DROP_RATE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)

        if self.output_stride == 8:
            strides = (1, 1)
            dilations = (2, 4)
        elif self.output_stride == 16:
            strides = (2, 1)
            dilations = (1, 2)
        elif self.output_stride == 32:
            strides = (2, 2)
            dilations = (1, 1)
        else:
            raise AssertionError

        bn_size = 4  # default
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', self.norm_layer(num_init_features)),
            ('relu0', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0 or i == 1:
                block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, self.drop_rate,
                                    padding=1, dilation=1, norm_layer=self.norm_layer)
            elif i == 2:
                block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, self.drop_rate,
                                    padding=dilations[0], dilation=dilations[0], norm_layer=self.norm_layer)
            elif i == 3:
                block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, self.drop_rate,
                                    padding=dilations[1], dilation=dilations[1], norm_layer=self.norm_layer)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                if i == 0:
                    trans = _Transition(num_features, num_features // 2, stride=2, norm_layer=self.norm_layer)
                elif i == 1:
                    trans = _Transition(num_features, num_features // 2, stride=strides[0], norm_layer=self.norm_layer)
                elif i == 2:
                    trans = _Transition(num_features, num_features // 2, stride=strides[1], norm_layer=self.norm_layer)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.num_features = num_features

        # Final batch norm
        self.features.add_module('norm5', self.norm_layer(num_features))
        self.dim_out = [None, None, None, self.num_features]

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, True)
        return [None, None, None, out]


@BACKBONE_REGISTRY.register()
def densenet121():
    num_init_features, growth_rate, block_config = densenet_cfg[121]
    return DenseNet(growth_rate, block_config, num_init_features)


@BACKBONE_REGISTRY.register()
def densenet161():
    num_init_features, growth_rate, block_config = densenet_cfg[161]
    return DenseNet(growth_rate, block_config, num_init_features)


@BACKBONE_REGISTRY.register()
def densenet169():
    num_init_features, growth_rate, block_config = densenet_cfg[169]
    return DenseNet(growth_rate, block_config, num_init_features)


@BACKBONE_REGISTRY.register()
def densenet201():
    num_init_features, growth_rate, block_config = densenet_cfg[201]
    return DenseNet(growth_rate, block_config, num_init_features)
