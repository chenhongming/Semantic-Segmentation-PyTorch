# modified from torchvision.models.resnet
import torch.nn as nn

from .build import BACKBONE_REGISTRY
from config.config import cfg
from utils.utils import set_norm

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, downsample=None):
        super(BasicBlock, self).__init__()
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

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
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
    def __init__(self, bottleneck=True, layers=(2, 2, 2, 2), base_width=64):
        super(ResNet, self).__init__()
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

        self.inplanes = base_width  # default 64
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

        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], 2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], strides[0], dilations[0])
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], strides[1], dilations[1])
        self.dim_out = [base_width * 4 * block.expansion, base_width * 8 * block.expansion]

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
        layers.append(block(self.inplanes, planes, stride, dilation, self.norm_layer, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilation, self.norm_layer))

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

        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)

        return [c3, c4]


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


if __name__ == '__main__':
    resnet50 = resnet50()
    print(resnet50)
