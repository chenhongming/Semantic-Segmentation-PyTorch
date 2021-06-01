import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.backbone.build import set_backbone
from models.model_zone import MODEL_REGISTRY
from utils.utils import set_norm

__all__ = ['PSPNet', 'psp']


class PyramidPoolingModule(nn.Module):

    def __init__(self, dim_in):
        super().__init__()
        ppm_hidden_dim = cfg.PPM.PPM_HIDDEN_DIM  # default: 512
        ppm_out_dim = cfg.PPM.PPM_OUT_DIM
        norm_layer = set_norm(cfg.MODEL.NORM_LAYER)

        self.dim_in = dim_in[-1]
        self.ppm = []
        for scale in cfg.PPM.POOL_SCALES:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(self.dim_in, ppm_hidden_dim, kernel_size=1, bias=False),
                norm_layer(ppm_hidden_dim),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.dim_in = self.dim_in + len(cfg.PPM.POOL_SCALES) * ppm_hidden_dim

        self.conv_last = nn.Sequential(
            nn.Conv2d(self.dim_in, ppm_out_dim, kernel_size=3, padding=1, bias=False),
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


class PSPNet(nn.Module):

    def __init__(self):
        super().__init__()
        dropout = cfg.PPM.DROP_OUT
        classes = cfg.DATA.CLASSES
        norm_layer = set_norm(cfg.MODEL.NORM_LAYER)
        self.zoom_factor = cfg.MODEL.ROOM_FACTOR
        assert self.zoom_factor in [1, 2, 4, 8]

        self.backbone = set_backbone()
        self.head = PyramidPoolingModule(self.backbone.dim_out)
        if cfg.PPM.USE_AUX and cfg.MODEL.PHASE == 'train':
            self.aux = nn.Sequential(
                nn.Conv2d(self.backbone.dim_out[0], self.head.dim_out, 3, padding=1, bias=False),
                norm_layer(self.head.dim_out),
                nn.ReLU(inplace=True))
        self.output = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(self.head.dim_out, classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.size()[2:]
        assert (size[0] - 1) % 8 == 0 and (size[1] - 1) % 8 == 0
        h = int((size[0] - 1) / 8 * self.zoom_factor + 1)
        w = int((size[1] - 1) / 8 * self.zoom_factor + 1)

        c3, c4 = self.backbone(x)
        c4 = self.head(c4)
        out = self.output(c4)
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        if cfg.PPM.USE_AUX and cfg.MODEL.PHASE == 'train':
            aux_out = self.aux(c3)
            aux_out = self.output(aux_out)
            if self.zoom_factor != 1:
                aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True)
            return out, aux_out
        return out


@MODEL_REGISTRY.register()
def psp():
    return PSPNet()
