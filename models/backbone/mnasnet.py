# modified from torchvision.models.mnasnet

import torch.nn as nn

from .build import BACKBONE_REGISTRY
from .op import _make_divisible
from config.config import cfg
from utils.utils import set_norm


__all__ = ['MNASNet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']


# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, expansion_factor,
                 bn_momentum=0.1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        padding = (kernel_size * dilation - dilation) // 2
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            norm_layer(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=mid_ch, bias=False),
            norm_layer(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, x):
        if self.apply_residual:
            return self.layers(x) + x
        else:
            return self.layers(x)


def _stack(in_ch, out_ch, kernel_size, stride, dilation, exp_factor, repeats,
           bn_momentum, norm_layer=nn.BatchNorm2d):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, dilation, exp_factor,
                              bn_momentum=bn_momentum, norm_layer=norm_layer)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, dilation, exp_factor,
                              bn_momentum=bn_momentum, norm_layer=norm_layer))
    return nn.Sequential(first, *remaining)


class MNASNet(nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha):
        super().__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        assert self._version in [1, 2]

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

        layers = [3, 3, 3, 2, 4, 1]
        num_of_channels = [32, 16, 24, 40, 80, 96, 192, 320]
        if self._version == 1:
            channels = [_make_divisible(ch * self.alpha) for ch in num_of_channels[2:]]
            channels.insert(0, 32)
            channels.insert(1, 16)
        else:
            channels = [_make_divisible(ch * self.alpha, 8) for ch in num_of_channels]

        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            self.norm_layer(channels[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1,
                      groups=channels[0], bias=False),
            self.norm_layer(channels[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, bias=False),
            self.norm_layer(channels[1], momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(channels[1], channels[2], kernel_size=3, exp_factor=3, repeats=layers[0],
                   stride=2, dilation=1, bn_momentum=_BN_MOMENTUM, norm_layer=self.norm_layer),
            _stack(channels[2], channels[3], kernel_size=5, exp_factor=3, repeats=layers[1],
                   stride=2, dilation=1, bn_momentum=_BN_MOMENTUM, norm_layer=self.norm_layer),
            _stack(channels[3], channels[4], kernel_size=5, exp_factor=6, repeats=layers[2],
                   stride=strides[0], dilation=dilations[0], bn_momentum=_BN_MOMENTUM, norm_layer=self.norm_layer),
            _stack(channels[4], channels[5], kernel_size=3, exp_factor=6, repeats=layers[3],
                   stride=1, dilation=dilations[0], bn_momentum=_BN_MOMENTUM, norm_layer=self.norm_layer),
            _stack(channels[5], channels[6], kernel_size=5, exp_factor=6, repeats=layers[4],
                   stride=strides[1], dilation=dilations[1], bn_momentum=_BN_MOMENTUM, norm_layer=self.norm_layer),
            _stack(channels[6], channels[7], kernel_size=3, exp_factor=6, repeats=layers[5],
                   stride=1, dilation=dilations[1], bn_momentum=_BN_MOMENTUM, norm_layer=self.norm_layer),
            # Final mapping to classifier input.
            nn.Conv2d(channels[7], 1280, 1, stride=1, padding=0, bias=False),
            self.norm_layer(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.dim_out = [None, None, None, 1280]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        return [None, None, None, x]


@BACKBONE_REGISTRY.register()
def mnasnet0_5():
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    return  MNASNet(alpha=0.5)


@BACKBONE_REGISTRY.register()
def mnasnet0_75():
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    return MNASNet(alpha=0.75)


@BACKBONE_REGISTRY.register()
def mnasnet1_0():
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    return MNASNet(alpha=1.0)


@BACKBONE_REGISTRY.register()
def mnasnet1_3():
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    return MNASNet(alpha=1.3)
