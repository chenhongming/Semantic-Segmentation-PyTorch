import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.backbone.build import set_backbone
from models.deeplabv3 import ASPP
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm

__all__ = ['DeepLabV3plus', 'deeplabv3plus']
# -------------------------------------------------------------------------------------- #
# supported backbone:
# 'mobilenet_v1' (multiplier=0.5, 1.0, 1.5, 2.0)
# 'resnet18', 'resnet34', 'resnet50', 'resnet101'(paper), 'resnet152',
# -------------------------------------------------------------------------------------- #


class DeepLabV3plusHead(nn.Module):

    def __init__(self, c2_channels, c5_channels, low_level_feature_channels, norm_layer):
        super().__init__()

        self.aspp = ASPP(in_channels=c5_channels)
        self.c2_block = nn.Sequential(
            nn.Conv2d(c2_channels, low_level_feature_channels, kernel_size=1, bias=False),
            norm_layer(low_level_feature_channels),
            nn.ReLU(True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(self.aspp.dim_out + low_level_feature_channels, 256, kernel_size=3, bias=False),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            norm_layer(256),
            nn.ReLU(True),
        )
        self.dim_out = 256  # default

    def forward(self, x):
        [c2, c5] = x
        size = c2.size()[2:]
        c5 = self.aspp(c5)
        c2 = self.c2_block(c2)
        x = F.interpolate(c5, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, c2], dim=1))


class DeepLabV3plus(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        self.zoom_factor = cfg.MODEL.ZOOM_FACTOR
        self.backbone_name = cfg.MODEL.BACKBONE_NAME
        self.dropout = cfg.DEEPLABV3PLUS.DROPOUT
        self.output_stride = cfg.DEEPLABV3PLUS.OUTPUT_STRIDE
        self.low_level_feature_channels = cfg.DEEPLABV3PLUS.LOW_LEVEL_FEATURE_CHANNELS
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        assert self.zoom_factor in [1, 2, 4, 8]

        if self.output_stride != 16:
            raise Exception("deeplabv3plus only supported output_stride == 16")
        if not self.backbone_name.startswith('resnet') or self.backbone_name.startswith('mobilenet_v1'):
            raise Exception("Unsupported backbone")
        self.backbone = set_backbone()
        self.head = DeepLabV3plusHead(self.backbone.dim_out[0], self.backbone.dim_out[-1],
                                      self.low_level_feature_channels, self.norm_layer)
        if cfg.DEEPLABV3PLUS.USE_AUX and cfg.MODEL.PHASE == 'train' and self.backbone.dim_out[-2] is not None:
            self.aux = nn.Sequential(
                nn.Conv2d(self.backbone.dim_out[-2], self.head.dim_out, 3, padding=1, bias=False),
                self.norm_layer(self.head.dim_out),
                nn.ReLU(inplace=True))
        self.output = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv2d(self.head.dim_out, self.classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.size()[2:]
        assert (size[0] - 1) % 8 == 0 and (size[1] - 1) % 8 == 0
        h = int((size[0] - 1) / 8 * self.zoom_factor + 1)
        w = int((size[1] - 1) / 8 * self.zoom_factor + 1)

        c2, _, c4, c5 = self.backbone(x)
        out = self.head([c2, c5])
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        if cfg.DEEPLABV3PLUS.USE_AUX and cfg.MODEL.PHASE == 'train' and c4 is not None:
            aux_out = self.aux(c4)
            aux_out = self.output(aux_out)
            if self.zoom_factor != 1:
                aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True)
            return out, aux_out
        return out


@MODEL_REGISTRY.register()
def deeplabv3plus():
    return DeepLabV3plus()
