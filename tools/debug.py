import math
import torch
import cv2 as cv
import torchvision
import torchvision.models as models

from config.config import cfg
from utils.plot import Writer

# model = models.mobilenet_v2()
# print(torch.__version__)
# print(torchvision.__version__)
#
# x = torch.rand([2, 3, 65, 65])
# o = model(x)
# print(o.size())

# writer = Writer(cfg.CKPT)
# writer.draw_curve(cfg.MODEL.NAME)
