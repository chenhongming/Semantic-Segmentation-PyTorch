import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.backbone.build import set_backbone
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm

__all__ = ['PSPNet', 'psp']
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


class PSPNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = cfg.PPM.DROP_OUT
        self.classes = cfg.DATA.CLASSES
        self.ppm_hidden_dim = cfg.PPM.PPM_HIDDEN_DIM  # default: 512
        self.ppm_out_dim = cfg.PPM.PPM_OUT_DIM
        self.norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.output_stride = cfg.MODEL.OUTPUT_STRIDE

        if cfg.MODEL.BACKBONE_NAME.startswith('vgg'):
            raise Exception("Not supported bankbone!")
        self.backbone = set_backbone()
        self.head = PyramidPoolingModule(self.backbone.dim_out[-1], self.ppm_hidden_dim,
                                         self.ppm_out_dim, self.norm_layer)
        if cfg.MODEL.USE_AUX and cfg.MODEL.PHASE == 'train' and self.backbone.dim_out[-2] is not None:
            self.aux = nn.Sequential(
                nn.Conv2d(self.backbone.dim_out[-2], self.head.dim_out, 3, padding=1, bias=False),
                self.norm_layer(self.head.dim_out),
                nn.ReLU(inplace=True))
        self.output = nn.Sequential(
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(self.head.dim_out, self.classes, kernel_size=1)
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
        if cfg.MODEL.PHASE == 'test':
            out = F.interpolate(out, size=x_size, mode='bilinear', align_corners=True)
        return out


class PyramidPoolingModule(nn.Module):

    def __init__(self, dim_in, ppm_hidden_dim, ppm_out_dim, norm_layer):
        super().__init__()

        self.ppm = []
        for scale in cfg.PPM.POOL_SCALES:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(dim_in, ppm_hidden_dim, kernel_size=1, bias=False),
                norm_layer(ppm_hidden_dim),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        dim_in = dim_in + len(cfg.PPM.POOL_SCALES) * ppm_hidden_dim

        self.conv_last = nn.Sequential(
            nn.Conv2d(dim_in, ppm_out_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(ppm_out_dim),
            nn.ReLU(inplace=True)
        )
        self.dim_out = ppm_out_dim

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.ppm:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        out = self.conv_last(out)
        return out


@MODEL_REGISTRY.register()
def psp():
    return PSPNet()
