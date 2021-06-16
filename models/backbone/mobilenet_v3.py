# modified from torchvision.models.mobilenetv3

import torch.nn as nn
from torch.nn import functional as F
from functools import partial

from .build import BACKBONE_REGISTRY
from .op import _make_divisible
from .mobilenet_v2 import ConvBNActivation
from config.config import cfg
from utils.utils import set_norm

__all__ = ['MobileNetV3', 'mobilenet_v3_large', 'mobilenet_v3_small']


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels, squeeze_factor=4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)

    def _scale(self, x, inplace):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, x):
        scale = self._scale(x, True)
        return scale * x


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, expanded_channels, kernel_size=3, stride=1, dilation=1,
                 use_hs='HS', use_se=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        activation_layer = nn.Hardswish if use_hs == 'HS' else nn.ReLU
        se_layer = SqueezeExcitation
        hidden_dim = _make_divisible(expanded_channels, 8)
        layers = []
        # expand
        if hidden_dim != in_channels:
            layers.append(ConvBNActivation(in_channels, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=activation_layer))
        # depthwise
        layers.append(ConvBNActivation(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                       groups=hidden_dim, norm_layer=norm_layer, activation_layer=activation_layer))
        if use_se:
            layers.append(se_layer(hidden_dim))

        # project
        layers.append(ConvBNActivation(hidden_dim, out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):

    def __init__(self, mode):
        super().__init__()

        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        if self.norm_layer == nn.BatchNorm2d:
            self.norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        self.multiplier = cfg.MODEL.MULTIPLIER
        assert mode in ['large', 'small']

        if self.output_stride == 8:
            dilations = (2, 4)
        elif self.output_stride == 16:
            dilations = (1, 2)
        elif self.output_stride == 32:
            dilations = (1, 1)
        else:
            raise AssertionError

        layers = []
        # building first layer
        input_channels = int(16 * self.multiplier if self.multiplier > 1.0 else 16)
        layers.append(ConvBNActivation(3, input_channels, kernel_size=3, stride=2,
                                       norm_layer=self.norm_layer, activation_layer=nn.Hardswish))
        if mode == 'small':
            inverted_residual_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, True, 'RE', 2],

                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],

                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],

                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
            s2_pos = [0, 1, 3, 8]
        else:
            inverted_residual_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],

                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],

                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],

                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],
            ]
            s2_pos = [0, 3, 6, 12]
        # building inverted residual blocks
        self.planes = input_channels
        layers.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[s2_pos[0]:s2_pos[1]],
                                       dilation=1))
        layers.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[s2_pos[1]:s2_pos[2]],
                                       dilation=1))
        layers.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[s2_pos[2]:s2_pos[3]],
                                       dilation=dilations[0]))
        layers.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[s2_pos[3]:],
                                       dilation=dilations[1]))
        # building last layers
        if mode == 'large':
            last_channels = int(960 * self.multiplier if self.multiplier > 1.0 else 960)
        else:
            last_channels = int(576 * self.multiplier if self.multiplier > 1.0 else 576)
        layers.append(ConvBNActivation(self.planes, last_channels, kernel_size=1, norm_layer=self.norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.dim_out = [None, None, None, last_channels]

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, inverted_residual_setting, dilation=1):
        features = list()
        for k, exp_size, c, se, nl, s in inverted_residual_setting:
            out_channels = int(c * self.multiplier)
            stride = s if dilation == 1 else 1
            features.append(block(planes, out_channels, expanded_channels=exp_size, kernel_size=k, stride=stride,
                                  dilation=dilation, use_hs=nl, use_se=se, norm_layer=self.norm_layer))
            planes = out_channels
        self.planes = planes
        return features

    def forward(self, x):
        x = self.features(x)
        return [None, None, None, x]


@BACKBONE_REGISTRY.register()
def mobilenet_v3_large():
    return MobileNetV3(mode='large')


@BACKBONE_REGISTRY.register()
def mobilenet_v3_small():
    return MobileNetV3(mode='small')