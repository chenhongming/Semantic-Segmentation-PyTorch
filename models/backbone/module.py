# copy and modifies from .mobilenet_v1.py for other model
import torch.nn as nn


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

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

    def __init__(self,  in_channels, out_channels, stride=1, padding=1, dilation=1, relu6=False,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                       groups=in_channels, relu6=relu6, norm_layer=norm_layer),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, relu6=relu6, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.conv(x)


# mobilenetv3
class ConvBNActivation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, activation_layer=None):
        padding = (kernel_size * dilation - dilation) // 2
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(out_channels),
            activation_layer(True),
        )
        self.out_channels = out_channels
