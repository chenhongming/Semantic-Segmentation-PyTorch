import torch
from torch import nn
from torch.nn import functional as F

from .backbone.build import set_backbone
from config.config import cfg
from models.backbone.module import ConvBNReLU
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm


__all__ = ['BiSeNet', 'bisenet']
# -------------------------------------------------------------------------------------- #
# supported backbone:
# 'mobilenet_v1' (multiplier=0.5, 1.0, 1.5, 2.0)
# 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
# 'shufflenet_v1_g1', 'shufflenet_v1_g2', 'shufflenet_v1_g3',
# 'shufflenet_v1_g4', 'shufflenet_v1_g8' (multiplier=0.5, 1.0, 1.5, 2.0)
# 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
# -------------------------------------------------------------------------------------- #


class BiSeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        self.in_channels = cfg.BISENET.IN_CHANNELS
        self.spatial_path_out_channels = cfg.BISENET.SPATIAL_PATH_OUT_CHANNELS
        self.context_path_out_channels = cfg.BISENET.CONTEXT_PATH_OUT_CHANNELS
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.dropout = cfg.BISENET.DROP_RATE
        self.aux = cfg.MODEL.USE_AUX

        if self.output_stride != 32:
            raise Exception("bisenet only supported output_stride == 32")
        if not (cfg.MODEL.BACKBONE_NAME.startswith('mobilenet_v1')
                or cfg.MODEL.BACKBONE_NAME.startswith('resnet')
                or cfg.MODEL.BACKBONE_NAME.startswith('shufflenet')):
            raise Exception("bisenet unsupported backbone")

        self.spatial_path = SpatialPath(self.in_channels, self.spatial_path_out_channels, norm_layer=self.norm_layer)
        self.context_path = ContextPath(self.in_channels, self.context_path_out_channels, norm_layer=self.norm_layer)
        self.ffm = FeatureFusionModule(self.spatial_path.dim_out + self.context_path.dim_out,
                                       self.spatial_path.dim_out + self.context_path.dim_out, norm_layer=self.norm_layer)
        self.head = BiSeHead(self.ffm.dim_out, self.classes, self.dropout, norm_layer=self.norm_layer)
        if self.aux:
            self.auxlayer1 = BiSeHead(self.context_path.backbone.dim_out[-2], self.classes, self.dropout,
                                      norm_layer=self.norm_layer)
            self.auxlayer2 = BiSeHead(self.context_path.backbone.dim_out[-1], self.classes, self.dropout,
                                      norm_layer=self.norm_layer)

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
        x_size = x.size()[2:]

        spatial_out = self.spatial_path(x)
        context_out, context_auxout = self.context_path(x)
        ffm_out = self.ffm([spatial_out, context_out])
        out = self.head(ffm_out)
        out = F.interpolate(out, size=x_size, mode='bilinear', align_corners=True)

        if self.aux and cfg.MODEL.PHASE == 'train':
            auxout1 = self.auxlayer1(context_auxout[0])
            auxout2 = self.auxlayer2(context_auxout[1])
            auxout1 = F.interpolate(auxout1, size=x_size, mode='bilinear', align_corners=True)
            auxout2 = F.interpolate(auxout2, size=x_size, mode='bilinear', align_corners=True)
            return out, auxout1, auxout2
        return out


class SpatialPath(nn.Module):
    """Spatial path"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inter_channels = 64
        self.conv1 = ConvBNReLU(in_channels, inter_channels, 7, 2, 3, norm_layer=norm_layer)
        self.conv2 = ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)
        self.conv3 = ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)
        self.conv4 = ConvBNReLU(inter_channels, out_channels, 1, 1, 0, norm_layer=norm_layer)
        self.dim_out = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class ContextPath(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.backbone = set_backbone()
        self.arm16 = AttentionRefinementModule(self.backbone.dim_out[-2], out_channels, norm_layer)
        self.arm32 = AttentionRefinementModule(self.backbone.dim_out[-1], out_channels, norm_layer)
        self.avg = GlobalAvgPooling(self.backbone.dim_out[-1], out_channels)
        self.dim_out = out_channels

    def forward(self, x):
        _, _, c4, c5 = self.backbone(x)
        aux_out = [c4, c5]

        c4 = self.arm16(c4)

        c5 = self.arm32(c5) + self.avg(c5)
        c5_up = F.interpolate(c5, size=c4.size()[2:], mode='bilinear', align_corners=True)

        c4 = c4 + c5_up
        size = (c4.size()[2] * 2, c4.size()[3] * 2)
        out = F.interpolate(c4, size=size, mode='bilinear', align_corners=True)
        return out, aux_out


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 1, 1, 0, norm_layer=norm_layer)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            norm_layer(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 1, 1, 0, norm_layer=norm_layer)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.dim_out = out_channels

    def forward(self, x):
        x1, x2 = x
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out


class GlobalAvgPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class BiSeHead(nn.Sequential):

    def __init__(self, in_channels, classes, dropout=0.1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            ConvBNReLU(in_channels, in_channels, 3, 1, 1, norm_layer=norm_layer),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, classes, 1)
        )


@MODEL_REGISTRY.register()
def bisenet():
    return BiSeNet()
