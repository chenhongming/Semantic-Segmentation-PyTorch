# modified from torchvision.models.resnet
import torch.nn as nn

from .build import BACKBONE_REGISTRY
from config.config import cfg
from utils.utils import set_norm

__all__ = ['MobileNetV2', 'mobilenet_v2']


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(out_channels),
            nn.ReLU6(True),
        )
        self.out_channels = out_channels


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels,  expand_ratio, stride=1, padding=0,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = list()
        hidden_dim = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, stride=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                       groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
            norm_layer(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, multiplier=1.0):
        super(MobileNetV2, self).__init__()
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.multiplier = cfg.MODEL.MULTIPLIER

        if self.output_stride == 8:
            dilations = (2, 4)
        elif self.output_stride == 16:
            dilations = (1, 2)
        elif self.output_stride == 32:
            dilations = (1, 1)
        else:
            raise AssertionError
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # building first layer
        input_channels = int(32 * self.multiplier if self.multiplier > 1.0 else 32)
        last_channels = int(1280 * self.multiplier if self.multiplier > 1.0 else 1280)
        self.features = [ConvBNReLU(3, input_channels, kernel_size=3, stride=2, padding=1, norm_layer=self.norm_layer)]
        # building inverted residual blocks
        self.planes = input_channels
        self.features.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[0:1]))
        self.features.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[1:2]))
        self.features.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[2:3]))
        self.features.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[3:5],
                                              dilation=dilations[0]))
        self.features.extend(self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[5:],
                                              dilation=dilations[1]))
        # building last layers
        self.features.append(ConvBNReLU(self.planes, last_channels, kernel_size=1, norm_layer=self.norm_layer))
        self.dim_out = [last_channels]
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

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
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.multiplier)
            for i in range(n):
                stride = s if i == 0 and dilation == 1 else 1
                features.append(block(planes, out_channels, t, stride=stride, dilation=dilation,
                                      padding=dilation, norm_layer=self.norm_layer))
                planes = out_channels
        self.planes = planes
        return features

    def forward(self, x):
        x = self.features(x)
        return x


@BACKBONE_REGISTRY.register()
def mobilenet_v2():
    return MobileNetV2()
