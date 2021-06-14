import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.backbone.build import set_backbone
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm


__all__ = ['FCN32s', 'FCN16s', 'FCN8s', 'fcn32s', 'fcn16s', 'fcn8s']
# -------------------------------------------------------------------------------------- #
# supported backbone:
# 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
# -------------------------------------------------------------------------------------- #


# vgg downsampling rate = 8, 16 layer pos
model_map = {
    'vgg11': [10, 15],
    'vgg11_bn': [14, 21],
    'vgg13': [14, 19],
    'vgg13_bn': [20, 27],
    'vgg16': [16, 23],
    'vgg16_bn': [23, 33],
    'vgg19': [18, 27],
    'vgg19_bn': [26, 39],
}


class FCNHead(nn.Sequential):

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            set_norm(cfg.MODEL.NORM_LAYER)(inter_channels),
            nn.ReLU(True),
            nn.Dropout(cfg.FCN.DROP_OUT),
            nn.Conv2d(inter_channels, channels, kernel_size=1)
        ]
        super().__init__(*layers)


class FCN32s(nn.Module):

    def __init__(self):
        super().__init__()
        #  cfg prams
        self.classes = cfg.DATA.CLASSES
        if not cfg.MODEL.BACKBONE_NAME.startswith('vgg'):
            raise Exception("fcn only supported vgg backbone!")

        self.backbone = set_backbone()
        self.head = FCNHead(self.backbone.dim_out, self.classes)

    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone(x)
        x = self.head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class FCN16s(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        if not cfg.MODEL.BACKBONE_NAME.startswith('vgg'):
            raise Exception("fcn only supported vgg backbone!")

        self.backbone = set_backbone()
        self.head = FCNHead(self.backbone.dim_out, self.classes)
        self.head_16s = nn.Conv2d(512, self.classes, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone.features[:model_map[cfg.MODEL.BACKBONE_NAME][1]+1](x)
        c4 = self.head_16s(x)

        x = self.backbone.features[model_map[cfg.MODEL.BACKBONE_NAME][1]+1:](x)
        c5 = self.head(x)
        c5_up = F.interpolate(c5, c4.size()[2:], mode='bilinear', align_corners=True)

        out = F.interpolate(c4+c5_up, size, mode='bilinear', align_corners=True)
        return out


class FCN8s(nn.Module):

    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        if not cfg.MODEL.BACKBONE_NAME.startswith('vgg'):
            raise Exception("fcn only supported vgg backbone!")

        self.backbone = set_backbone()
        self.head = FCNHead(self.backbone.dim_out, self.classes)
        self.head_8s = nn.Conv2d(256, self.classes, kernel_size=1)
        self.head_16s = nn.Conv2d(512, self.classes, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone.features[:model_map[cfg.MODEL.BACKBONE_NAME][0] + 1](x)
        c3 = self.head_8s(x)

        x = self.backbone.features[model_map[cfg.MODEL.BACKBONE_NAME][0] + 1:
                                   model_map[cfg.MODEL.BACKBONE_NAME][1] + 1](x)
        c4 = self.head_16s(x)

        x = self.backbone.features[model_map[cfg.MODEL.BACKBONE_NAME][1] + 1:](x)
        c5 = self.head(x)
        c5_up = F.interpolate(c5, c4.size()[2:], mode='bilinear', align_corners=True)

        c4_up = F.interpolate(c4+c5_up, c4.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(c3+c4_up, size, mode='bilinear', align_corners=True)
        return out


@MODEL_REGISTRY.register()
def fcn32s():
    return FCN32s()


@MODEL_REGISTRY.register()
def fcn16s():
    return FCN16s()


@MODEL_REGISTRY.register()
def fcn8s():
    return FCN8s()
