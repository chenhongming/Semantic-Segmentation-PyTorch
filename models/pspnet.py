import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg


class PyramidPoolingModule(nn.Module):
    def __init__(self, dim_in, norm_layer=nn.BatchNorm2d):
        super().__init__()
        ppm_dim = cfg.PPM.PPM_DIM  # default: 512

        self.dim_in = dim_in
        self.ppm = []
        for scale in cfg.PPM.POOL_SCALES:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(self.dim_in, ppm_dim, kernel_size=1, bias=False),
                norm_layer(ppm_dim),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.dim_in = self.dim_in + len(cfg.SEMSEG.PPM.POOL_SCALES) * ppm_dim

        self.conv_last = nn.Sequential(
            nn.Conv2d(self.dim_in, 512, kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True)
        )

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

