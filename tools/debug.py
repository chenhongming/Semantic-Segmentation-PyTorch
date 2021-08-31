import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import cv2 as cv
import torchvision
import torchvision.models as models

import _init_path
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

# capture = cv.VideoCapture(0)
# while capture.isOpened():
#     ret, frame = capture.read()
#     cv.imshow("frame", frame)
#     if cv.waitKey(1) & 0xFF == ord("q"):
#         break
# capture.release()
# cv.destroyAllWindows()

# dist.init_process_group(backend="nccl", init_method='env://')
num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 0
print(num_gpus)
