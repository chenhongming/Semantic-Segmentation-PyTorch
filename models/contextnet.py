from torch import nn
from torch.nn import functional as F

from config.config import cfg
from models.backbone.module import ConvBNReLU, DWConvBNReLU
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm

__all__ = ["ContextNet", 'contextnet']
# -------------------------------------------------------------------------------------- #
# supported backbone: None
# !!! contextnet is not needed backbone !!!
# -------------------------------------------------------------------------------------- #


class ContextNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        self.channels = cfg.ContextNet.CHANNELS
        self.expansion_factor = cfg.ContextNet.EXPANSION_FACTOR
        self.layers = cfg.ContextNet.LAYERS
        self.in_channels = cfg.ContextNet.IN_CHANNELS
        self.hidden_channels1 = cfg.ContextNet.HIDDEN_CHANNELS1
        self.hidden_channels2 = cfg.ContextNet.HIDDEN_CHANNELS2
        self.out_channels = cfg.ContextNet.OUT_CHANNELS
        self.aux = cfg.MODEL.USE_AUX
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        if cfg.MODEL.BACKBONE_NAME is not None:
            raise Exception("contextnet is not needed backbone")
        if self.output_stride != 8:
            raise Exception("contextnet only supported output_stride == 8")

        self.shadow_net = ShallowNet(self.in_channels, self.hidden_channels1, self.hidden_channels2,
                                     self.out_channels, self.norm_layer)
        self.deep_net = DeepNet(self.in_channels, self.hidden_channels1, self.channels, self.expansion_factor,
                                self.layers, self.norm_layer)
        self.feature_fusion_module = FeatureFusionModule(self.shadow_net.dim_out, self.deep_net.dim_out,
                                                         self.out_channels, self.norm_layer)
        self.output = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(self.feature_fusion_module.dim_out, self.classes, kernel_size=1)
        )
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(self.deep_net.dim_out, 32, kernel_size=3, padding=1, bias=False),
                self.norm_layer(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, self.classes, kernel_size=1)
            )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_size = x.size()[2:]  # for test
        size = (x.size()[2] // 4, x.size()[3] // 4)
        s = self.shadow_net(x)

        x = self.deep_net(F.interpolate(x, size=size, mode='bilinear', align_corners=True))
        out = self.feature_fusion_module([s, x])
        out = self.output(out)
        out = F.interpolate(out, size=x_size, mode='bilinear', align_corners=True)
        if self.aux and cfg.MODEL.PHASE == 'train':
            auxout = self.auxlayer(x)
            auxout = F.interpolate(auxout, size=x_size, mode='bilinear', align_corners=True)
            return out, auxout
        return out


class ShallowNet(nn.Module):

    def __init__(self, in_channels, hidden_channels1=32, hidden_channels2=64, out_channels=128, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, hidden_channels1, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer)
        self.conv2 = DWConvBNReLU(hidden_channels1, hidden_channels2, stride=2, norm_layer=norm_layer)
        self.conv3 = DWConvBNReLU(hidden_channels2, out_channels, stride=2, norm_layer=norm_layer)
        self.conv4 = DWConvBNReLU(out_channels, out_channels, stride=1, norm_layer=norm_layer)
        self.dim_out = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DeepNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, block_channels, t, num_blocks, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.block_channels = block_channels
        self.t = t
        self.num_blocks = num_blocks
        self.norm_layer = norm_layer

        # the first layer
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1,
                                norm_layer=self.norm_layer)
        # building layers
        self.inplanes = hidden_channels
        self.bottleneck1 = self._make_layer(LinearBottleneck, block_channels[0], num_blocks[0], t[0], 1)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[1], num_blocks[1], t[1], 1)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[2], num_blocks[2], t[2], 2)
        self.bottleneck4 = self._make_layer(LinearBottleneck, block_channels[3], num_blocks[3], t[3], 2)
        self.bottleneck5 = self._make_layer(LinearBottleneck, block_channels[4], num_blocks[4], t[4], 1)
        self.bottleneck6 = self._make_layer(LinearBottleneck, block_channels[5], num_blocks[5], t[5], 1)
        self.dim_out = block_channels[5]

    def _make_layer(self, block, planes, blocks, t, stride=1):
        layers = [block(self.inplanes, planes, t, stride, self.norm_layer)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, t, 1, self.norm_layer))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        return x


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, in_channels * t, kernel_size=1, stride=1, norm_layer=norm_layer),
            DWConvBNReLU(in_channels * t, out_channels, stride=stride, norm_layer=norm_layer),
            )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class FeatureFusionModule(nn.Module):

    def __init__(self, higher_in_channels, lower_in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        #  deepnet branch
        self.branch4 = nn.Sequential(
            nn.Conv2d(lower_in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      groups=lower_in_channels, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            norm_layer(out_channels)
        )
        # shadow branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(higher_in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels)
        )
        self.relu = nn.ReLU(True)
        self.dim_out = out_channels

    def forward(self, x):
        shadow_feature, deepnet_feature = x
        size = shadow_feature.size()[2:]
        deepnet_feature_up = F.interpolate(deepnet_feature, size=size, mode='bilinear', align_corners=True)

        out = self.branch4(deepnet_feature_up) + self.branch1(shadow_feature)
        return self.relu(out)


@MODEL_REGISTRY.register()
def contextnet():
    return ContextNet()
