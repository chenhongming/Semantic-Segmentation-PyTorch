import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.backbone.build import set_backbone
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm

__all__ = ['LRASPP', 'lraspp']
# -------------------------------------------------------------------------------------- #
# supported backbone:
# 'mobilenet_v1' (multiplier=0.5, 1.0, 1.5, 2.0)
# 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
# -------------------------------------------------------------------------------------- #


class LRASPP(nn.Module):
    """
    Implement a Lite R-ASPP Network for semantic segmentation from
    "Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>
    """
    def __init__(self):
        super().__init__()

        self.classes = cfg.DATA.CLASSES
        self.zoom_factor = cfg.MODEL.ZOOM_FACTOR
        self.backbone_name = cfg.MODEL.BACKBONE_NAME
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.inter_channels = cfg.LRASPP.INTER_CHANNELS
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        assert self.zoom_factor in [1, 2, 4, 8]
        if not (self.backbone_name.startswith('resnet') or self.backbone_name.startswith('mobilenet_v1')):
            raise Exception("Unsupported backbone")

        self.backbone = set_backbone()
        self.head = LRASPPHead(self.backbone.dim_out[0], self.backbone.dim_out[-1],
                               self.inter_channels, self.classes, self.norm_layer)

    def forward(self, x):
        out_size = (x.size()[2] // self.output_stride, x.size()[3] // self.output_stride)

        c2, _, _, c5 = self.backbone(x)
        out = self.head([c2, c5])
        out = F.interpolate(out, out_size, mode='bilinear', align_corners=True)
        return out


class LRASPPHead(nn.Module):
    """
    Args:
        c2_channels (int): the number of channels of the c2(low) level features of backbone.
        c5_channels (int): the number of channels of the c5(high) level features of backbone.
        inter_channels (int): the number of channels for intermediate computations.
    """

    def __init__(self, c2_channels, c5_channels, inter_channels, classes, norm_layer):
        super().__init__()

        self.cbr = nn.Sequential(
            nn.Conv2d(c5_channels, inter_channels, kernel_size=3, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True))
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c5_channels, inter_channels, kernel_size=1, bias=False),
            nn.Sigmoid())
        self.c2_classifier = nn.Conv2d(c2_channels, classes, kernel_size=1, bias=False)
        self.c5_classifier = nn.Conv2d(inter_channels, classes, kernel_size=1, bias=False)

    def forward(self, x):
        c2, c5 = x
        x = self.cbr(c5)
        s = self.scale(c5)
        x = x * s
        x = F.interpolate(x, c2.size()[2:], mode='bilinear', align_corners=True)
        return self.c2_classifier(c2) + self.c5_classifier(x)


@MODEL_REGISTRY.register()
def lraspp():
    return LRASPP()
