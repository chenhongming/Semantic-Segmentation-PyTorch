# modified from torchvision.models.shufflenetv2
import torch
import torch.nn as nn

from .build import BACKBONE_REGISTRY
from config.config import cfg
from utils.utils import set_norm


__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]


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

    def __init__(self, inp, oup, stride, padding, dilation, branch1_flag=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.branch1_flag = branch1_flag
        self.norm_layer = norm_layer

        branch_features = oup // 2

        if self.stride > 1 or self.branch1_flag:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation),
                self.norm_layer(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                self.norm_layer(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1 or self.branch1_flag) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            self.norm_layer(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride,
                                padding=self.padding, dilation=self.dilation),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            self.norm_layer(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        return nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, bias=bias, groups=i)

    def forward(self, x):
        if self.stride > 1 or self.branch1_flag:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):

    def __init__(self, multiplier):
        super().__init__()
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.multiplier = multiplier
        self.stages_repeats = [4, 8, 4]
        self.stages_out_channels = {
            0.5: [24, 48, 96, 192, 1024],   1.0: [24, 116, 232, 464, 1024],
            1.5: [24, 176, 352, 704, 1024], 2.0: [24, 244, 488, 976, 2048],
        }
        self.channels = self.stages_out_channels[multiplier]

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
            nn.Conv2d(3, self.channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = self.channels[0]
        # make stage
        self.stage2 = self._make_stage(InvertedResidual, self.channels[1], self.stages_repeats[0],
                                       stride=2, dilation=1)
        self.stage3 = self._make_stage(InvertedResidual, self.channels[2], self.stages_repeats[1],
                                       stride=strides[0], dilation=dilations[0])
        self.stage4 = self._make_stage(InvertedResidual, self.channels[3], self.stages_repeats[2],
                                       stride=strides[1], dilation=dilations[1])

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.channels[-1], kernel_size=1, stride=1, padding=0, bias=False),
            self.norm_layer(self.channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.dim_out = [None, self.channels[1], self.channels[2], self.channels[-1]]

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
        branch1_flag = False if stride == 2 else True
        layers = list()
        layers.append(block(self.inplanes, planes, stride=stride, padding=dilation,
                            dilation=dilation, norm_layer=self.norm_layer, branch1_flag=branch1_flag))
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, padding=dilation,
                                dilation=dilation, norm_layer=self.norm_layer))
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        c5 = self.conv5(c5)
        return [None, c3, c4, c5]


@BACKBONE_REGISTRY.register()
def shufflenet_v2_x0_5():
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return ShuffleNetV2(multiplier=0.5)


@BACKBONE_REGISTRY.register()
def shufflenet_v2_x1_0():
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return ShuffleNetV2(multiplier=1.0)


@BACKBONE_REGISTRY.register()
def shufflenet_v2_x1_5():
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return ShuffleNetV2(multiplier=1.5)


@BACKBONE_REGISTRY.register()
def shufflenet_v2_x2_0():
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return ShuffleNetV2(multiplier=2.0)
