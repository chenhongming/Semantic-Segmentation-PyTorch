# modified from torchvision.models.resnet

import torch.nn as nn

from .build import BACKBONE_REGISTRY
from config.config import cfg
from utils.utils import set_norm

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1,
                 base_width=64, norm_layer=nn.BatchNorm2d, downsample=None):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1,
                 base_width=64, norm_layer=nn.BatchNorm2d, downsample=None):
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, bottleneck=True, layers=(2, 2, 2, 2), groups=1, width_per_group=64):
        super().__init__()
        norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        head7x7 = cfg.MODEL.HEAD7X7
        if bottleneck:
            block = Bottleneck
        else:
            block = BasicBlock

        if output_stride == 8:
            strides = (1, 1)
            dilations = (2, 4)
        elif output_stride == 16:
            strides = (2, 1)
            dilations = (1, 2)
        elif output_stride == 32:
            strides = (2, 2)
            dilations = (1, 1)
        else:
            raise AssertionError

        self.inplanes = 64  # default 64
        self.groups = groups
        self.base_width = width_per_group
        self.head7x7 = head7x7
        self.norm_layer = norm_layer
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, bias=False)
            self.bn2 = norm_layer(self.inplanes)
            self.conv3 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, bias=False)
            self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64 * 2, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 64 * 4, layers[2], stride=strides[0], dilation=dilations[0])
        self.layer4 = self._make_layer(block, 64 * 8, layers[3], stride=strides[1], dilation=dilations[1])
        self.dim_out = [64 * block.expansion, 64 * 2 * block.expansion,
                        64 * 4 * block.expansion, 64 * 8 * block.expansion]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        Args:
            block: block type used to construct ResNet
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, self.groups, self.base_width, self.norm_layer,
                            downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilation, self.groups, self.base_width, self.norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]


@BACKBONE_REGISTRY.register()
def resnet18():
    """
    Constructs a ResNet-18 model.
    Args:
        bottleneck: Use bottleneck for building resnet model if Ture, otherwise use basicblock
        layers: Number of layers to build resnet model
    """
    return ResNet(bottleneck=False, layers=(2, 2, 2, 2))


@BACKBONE_REGISTRY.register()
def resnet34():
    """
    Constructs a ResNet-34 model.
    Args:
        bottleneck: Use bottleneck for building resnet model if Ture, otherwise use basicblock
        layers: Number of layers to build resnet model
    """
    return ResNet(bottleneck=False, layers=(3, 4, 6, 3))


@BACKBONE_REGISTRY.register()
def resnet50():
    """
    Constructs a ResNet-50 model.
    Args:
        bottleneck: Use bottleneck for building resnet model if Ture, otherwise use basicblock
        layers: Number of layers to build resnet model
    """
    return ResNet(bottleneck=True, layers=(3, 4, 6, 3))


@BACKBONE_REGISTRY.register()
def resnet101():
    """
    Constructs a ResNet-101 model.
    Args:
        bottleneck: Use bottleneck for building resnet model if Ture, otherwise use basicblock
        layers: Number of layers to build resnet model
    """
    return ResNet(bottleneck=True, layers=(3, 4, 23, 3))


@BACKBONE_REGISTRY.register()
def resnet152():
    """
    Constructs a ResNet-152 model.
    Args:
        bottleneck: Use bottleneck for building resnet model if Ture, otherwise use basicblock
        layers: Number of layers to build resnet model
    """
    return ResNet(bottleneck=True, layers=(3, 8, 36, 3))


@BACKBONE_REGISTRY.register()
def resnext50_32x4d():
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    groups = 32
    width_per_group = 4
    return ResNet(bottleneck=True, layers=(3, 4, 6, 3), groups=groups, width_per_group=width_per_group)


@BACKBONE_REGISTRY.register()
def resnext101_32x8d():
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    groups = 32
    width_per_group = 8
    return ResNet(bottleneck=True, layers=(3, 4, 23, 3), groups=groups, width_per_group=width_per_group)


@BACKBONE_REGISTRY.register()
def wide_resnet50_2():
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    width_per_group = 64 * 2
    return ResNet(bottleneck=True, layers=(3, 4, 6, 3), width_per_group=width_per_group)


@BACKBONE_REGISTRY.register()
def wide_resnet101_2():
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    width_per_group = 64 * 2
    return ResNet(bottleneck=True, layers=(3, 4, 23, 3), width_per_group=width_per_group)
