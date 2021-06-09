import torch
import torch.nn as nn

from .build import BACKBONE_REGISTRY
from config.config import cfg
from utils.utils import set_norm

__all__ = ['ShufflenetV1', 'shufflenet_v1_g1', 'shufflenet_v1_g2',
           'shufflenet_v1_g3', 'shufflenet_v1_g4', 'shufflenet_v1_g8']


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % groups == 0
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, padding, dilation, groups, shortcut_flag=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.shortcut_flag = shortcut_flag
        self.norm_layer = norm_layer

        hidden_channel = oup // 4
        g = 1 if inp == 24 else self.groups

        if self.stride > 1 or self.shortcut_flag:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=self.stride, padding=1)
            oup -= inp
        self.conv1 = nn.Conv2d(inp, hidden_channel, kernel_size=1, stride=1, padding=0, groups=g, bias=False)
        self.bn1 = self.norm_layer(hidden_channel)
        self.conv2 = self.depthwise_conv(hidden_channel, hidden_channel, kernel_size=3, stride=self.stride,
                                         padding=self.padding, dilation=self.dilation, groups=self.groups)
        self.bn2 = self.norm_layer(hidden_channel)
        self.conv3 = nn.Conv2d(hidden_channel, oup, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn3 = self.norm_layer(oup)
        self.relu = nn.ReLU(True)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        return nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = channel_shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.stride > 1 or self.shortcut_flag:
            t = self.shortcut(x)
            out = torch.cat((self.shortcut(x), out), dim=1)
        else:
            out = out + x
        out = self.relu(out)
        return out


class ShufflenetV1(nn.Module):

    def __init__(self, groups):
        super().__init__()
        assert groups in [1, 2, 3, 4, 8]

        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.multiplier = cfg.MODEL.MULTIPLIER
        self.stages_repeats = [4, 8, 4]
        self.stages_out_channels = {
            1: [144, 288, 576], 2: [200, 400, 800],
            3: [240, 480, 960], 4: [272, 544, 1088],
            8: [384, 768, 1536]
        }
        self.groups = groups
        self.channels = [int(ch * self.multiplier) for ch in self.stages_out_channels[self.groups]]

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

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # make stage
        self.inplanes = 24
        self.stage2 = self._make_stage(InvertedResidual, self.channels[0], self.stages_repeats[0],
                                       stride=2, dilation=1)
        self.stage3 = self._make_stage(InvertedResidual, self.channels[1], self.stages_repeats[1],
                                       stride=strides[0], dilation=dilations[0])
        self.stage4 = self._make_stage(InvertedResidual, self.channels[2], self.stages_repeats[2],
                                       stride=strides[1], dilation=dilations[1])
        self.dim_out = [self.channels[1], self.channels[-1]]

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, block, planes, blocks, stride=1, dilation=1):
        shortcut_flag = False if stride == 2 else True
        layers = list()
        layers.append(block(self.inplanes, planes, stride=stride, padding=dilation, dilation=dilation,
                            groups=self.groups, norm_layer=self.norm_layer, shortcut_flag=shortcut_flag))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, padding=dilation, dilation=dilation,
                                groups=self.groups, norm_layer=self.norm_layer))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        return [c3, c4]


@BACKBONE_REGISTRY.register()
def shufflenet_v1_g1():
    return ShufflenetV1(groups=1)


@BACKBONE_REGISTRY.register()
def shufflenet_v1_g2():
    return ShufflenetV1(groups=2)


@BACKBONE_REGISTRY.register()
def shufflenet_v1_g3():
    return ShufflenetV1(groups=3)


@BACKBONE_REGISTRY.register()
def shufflenet_v1_g4():
    return ShufflenetV1(groups=4)


@BACKBONE_REGISTRY.register()
def shufflenet_v1_g8():
    return ShufflenetV1(groups=8)
