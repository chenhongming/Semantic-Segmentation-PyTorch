import torch.nn as nn

from .build import BACKBONE_REGISTRY
from config.config import cfg
from utils.utils import set_norm


__all__ = ['MobileNetV1', 'mobilenet_v1']


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, norm_layers=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = norm_layers(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConvBNReLU(nn.Module):
    """
    Depthwise Separable Convolution in MobileNetV1
    depthwise convolution + pointwise convolution
    """

    def __init__(self,  in_channels, out_channels, stride=1, padding=0, dilation=1, norm_layers=nn.BatchNorm2d):
        super(DWConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                       groups=in_channels, norm_layers=norm_layers),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, norm_layers=norm_layers)
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetV1(nn.Module):

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.multiplier = cfg.MODEL.MULTIPLIER

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

        in_channel = int(32 * self.multiplier if self.multiplier > 1.0 else 32)
        channels = [int(ch * self.multiplier) for ch in [64, 128, 256, 512, 1024]]
        layers = [2, 2, 6, 2]
        self.conv1 = ConvBNReLU(3, in_channel, kernel_size=3, stride=2, padding=1, norm_layers=self.norm_layer)
        self.conv2 = DWConvBNReLU(in_channel, channels[0], padding=1, dilation=1)

        # building layers
        self.inplanes = channels[0]
        self.layer1 = self._make_layers(DWConvBNReLU, channels[1], layers[0], stride=2)
        self.layer2 = self._make_layers(DWConvBNReLU, channels[2], layers[1], stride=2)
        self.layer3 = self._make_layers(DWConvBNReLU, channels[3], layers[2], stride=strides[0], dilation=dilations[0])
        self.layer4 = self._make_layers(DWConvBNReLU, channels[4], layers[3], stride=strides[1], dilation=dilations[1])

        self.dim_out = [int(ch * self.multiplier) for ch in [512, 1024]]
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layers(self, block, planes, blocks, stride=1, dilation=1):
        layers = list()
        layers.append(block(self.inplanes, planes, stride=stride, padding=dilation,
                            dilation=dilation, norm_layers=self.norm_layer))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, padding=dilation,
                                dilation=dilation, norm_layers=self.norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c3, c4


@BACKBONE_REGISTRY.register()
def mobilenet_v1():
    return MobileNetV1()
