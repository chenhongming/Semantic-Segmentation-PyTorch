from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.backbone.build import set_backbone
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm

__all__ = ['DenseASPP', 'denseaspp']
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


class DenseASPP(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        self.backbone_name = cfg.MODEL.BACKBONE_NAME
        self.output_stride = cfg.DENSEASPP.OUTPUT_STRIDE
        self.inter_channels = cfg.DENSEASPP.INTER_CHANNELS
        self.out_channels = cfg.DENSEASPP.OUT_CHANNELS
        self.atrous_rate = cfg.DENSEASPP.ATROUS_RATE
        self.drop_rate = cfg.DENSEASPP.DROP_RATE
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)

        if cfg.MODEL.BACKBONE_NAME.startswith('vgg'):
            raise Exception("Not supported bankbone!")
        self.backbone = set_backbone()
        self.head = DenseASPPHead(self.backbone.dim_out[-1], self.inter_channels, self.out_channels,
                                  self.atrous_rate, self.drop_rate, self.norm_layer)
        if cfg.MODEL.USE_AUX and cfg.MODEL.PHASE == 'train' and self.backbone.dim_out[-2] is not None:
            self.aux = nn.Sequential(
                nn.Conv2d(self.backbone.dim_out[-2], self.head.dim_out, 3, padding=1, bias=False),
                self.norm_layer(self.head.dim_out),
                nn.ReLU(inplace=True))
        self.output = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(self.head.dim_out, self.classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x_size = x.size()[2:]  # for test
        out_size = (x.size()[2] // self.output_stride, x.size()[3] // self.output_stride)  # for train or val

        _, _, c4, c5 = self.backbone(x)
        c5 = self.head(c5)
        out = self.output(c5)
        if cfg.MODEL.USE_AUX and cfg.MODEL.PHASE == 'train' and c4 is not None:
            aux_out = self.aux(c4)
            aux_out = self.output(aux_out)
            aux_out = F.interpolate(aux_out, size=out_size, mode='bilinear', align_corners=True)
            return out, aux_out
        elif cfg.MODEL.PHASE == 'val':
            out = F.interpolate(out, size=out_size, mode='bilinear', align_corners=True)
        elif cfg.MODEL.PHASE == 'test':
            out = F.interpolate(out, size=x_size, mode='bilinear', align_corners=True)
        return out


class DenseASPPConv(nn.Sequential):

    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate, drop_rate=0.1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, kernel_size=1)),
        self.add_module('bn1', norm_layer(inter_channels)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module("conv2", nn.Conv2d(inter_channels, out_channels, kernel_size=3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels)),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super().forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class DenseASPPHead(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate, drop_rate=0.1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.aspp_3 = DenseASPPConv(in_channels, inter_channels, out_channels,
                                    atrous_rate[0], drop_rate, norm_layer)
        self.aspp_6 = DenseASPPConv(in_channels + out_channels * 1, inter_channels, out_channels,
                                    atrous_rate[1], drop_rate, norm_layer)
        self.aspp_12 = DenseASPPConv(in_channels + out_channels * 2, inter_channels, out_channels,
                                     atrous_rate[2], drop_rate, norm_layer)
        self.aspp_18 = DenseASPPConv(in_channels + out_channels * 3, inter_channels, out_channels,
                                     atrous_rate[3], drop_rate, norm_layer)
        self.aspp_24 = DenseASPPConv(in_channels + out_channels * 4, inter_channels, out_channels,
                                     atrous_rate[4], drop_rate, norm_layer)

        self.dim_out = in_channels + out_channels * 5

    def forward(self, x):
        aspp_3 = self.aspp_3(x)
        x = torch.cat([aspp_3, x], dim=1)

        aspp_6 = self.aspp_6(x)
        x = torch.cat([aspp_6, x], dim=1)

        aspp_12 = self.aspp_12(x)
        x = torch.cat([aspp_12, x], dim=1)

        aspp_18 = self.aspp_18(x)
        x = torch.cat([aspp_18, x], dim=1)

        aspp_24 = self.aspp_24(x)
        x = torch.cat([aspp_24, x], dim=1)

        return x


@MODEL_REGISTRY.register()
def denseaspp():
    return DenseASPP()
