import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.backbone.build import set_backbone
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm

__all__ = ['DeepLabV3', 'deeplabv3', 'ASPP']
# -------------------------------------------------------------------------------------- #
# supported backbone:
# 'mobilenet_v1' (multiplier=0.5, 1.0, 1.5, 2.0)
# 'mobilenet_v2' (multiplier=0.5, 1.0, 1.5, 2.0)
# 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'
# 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
# 'shufflenet_v1_g1', 'shufflenet_v1_g2', 'shufflenet_v1_g3',
# 'shufflenet_v1_g4', 'shufflenet_v1_g8' (multiplier=0.5, 1.0, 1.5, 2.0)
# 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
# 'densenet121', 'densenet161', 'densenet169', 'densenet201'
# -------------------------------------------------------------------------------------- #


class DeepLabV3(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.out_channels = cfg.ASPP.OUT_CHANNELS  # default 512
        self.dropout = cfg.ASPP.DROPOUT
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)

        if cfg.MODEL.BACKBONE_NAME.startswith('vgg'):
            raise Exception("Not supported bankbone!")
        self.backbone = set_backbone()
        self.head = ASPP(self.backbone.dim_out[-1], self.out_channels, self.output_stride, self.norm_layer)
        if cfg.MODEL.USE_AUX and cfg.MODEL.PHASE == 'train' and self.backbone.dim_out[-2] is not None:
            self.aux = nn.Sequential(
                nn.Conv2d(self.backbone.dim_out[-2], self.head.dim_out, kernel_size=3, padding=1, bias=False),
                self.norm_layer(self.head.dim_out),
                nn.ReLU(inplace=True))
        self.output = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv2d(self.head.dim_out, self.classes, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()[2:]

        _, _, c4, c5 = self.backbone(x)
        c5 = self.head(c5)
        out = self.output(c5)
        out = F.interpolate(out, size=x_size, mode='bilinear', align_corners=True)
        if cfg.MODEL.USE_AUX and cfg.MODEL.PHASE == 'train' and c4 is not None:
            aux_out = self.aux(c4)
            aux_out = self.output(aux_out)
            aux_out = F.interpolate(aux_out, size=x_size, mode='bilinear', align_corners=True)
            return out, aux_out
        return out


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation, norm_layer=nn.BatchNorm2d):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x_size = x.size()[2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, x_size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, output_stride, norm_layer):
        super().__init__()

        if output_stride == 8:
            atrous_rate = [12, 24, 36]
        elif output_stride == 16:
            atrous_rate = [6, 12, 18]
        else:
            raise Exception("output_stride only supported 8 or 16!")
        modules = list()
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        ))
        for rate in tuple(atrous_rate):
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.porject = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
        )
        self.dim_out = out_channels

    def forward(self, x):
        res = list()
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.porject(res)


@MODEL_REGISTRY.register()
def deeplabv3():
    return DeepLabV3()
