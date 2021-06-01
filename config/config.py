import os
from yacs.config import CfgNode as CN
from utils.utils import root_path

# ----------------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------------- #
"""
config system:
This file specifies default config options. You should not change values in this file. 
Instead, you should write a config file (in yaml) and use merge_cfg_from_file(yaml_file) 
to load it and override the default options.
"""

_C = CN()
cfg = _C
# ---------------------------------------------------------------------------- #
# DATA options
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.DATASET = 'ade20k'
_C.DATA.TRAIN_JSON = ""
_C.DATA.VAL_JSON = ""
_C.DATA.TEST_JSON = ""
_C.DATA.CLASSES = 150
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------- #
# TRAIN options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.AUGMENTATIONS = ['RandomFlip', 'RandomResize', 'RandomCrop', 'RandomRotate']
# Probability of using RandomFlip
_C.TRAIN.PROB = 0.5
# Ratio of using RandomResize
_C.TRAIN.RATIO = (0.5, 2)
# crop_size of using RandomCrop
# int: a square crop (crop_size, crop_size) is made
# list: [640, 512]
_C.TRAIN.CROP_SIZE = 512
# prams of using RandomRotate: PADDING for img, IGNORE_LABEL for mask
_C.TRAIN.ROTATE = (-10, 10)
_C.TRAIN.PADDING = (0, 0, 0)
_C.TRAIN.IGNORE_LABEL = 255

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.BACKBONE_NAME = 'resnet18'
_C.MODEL.BACKBONE_PRETRAINED = True
_C.MODEL.BACKBONE_WEIGHT = "pretrained/"
_C.MODEL.NORM_LAYER = 'bn'  # bn or syncbn
_C.MODEL.OUTPUT_STRIDE = 8  # 8, 16, 32
_C.MODEL.HEAD7X7 = False  # only for resnet backbone
_C.MODEL.PHASE = 'train'
_C.MODEL.RESUME = False
_C.MODEL.FINETUNE = False
_C.MODEL.NAME = 'psp'
_C.MODEL.PRETRAINED = True
_C.MODEL.MODEL_WEIGHT = "ckpts/ade20k/model.pth"
# output.size * room_factor
_C.MODEL.ROOM_FACTOR = 8
_C.MODEL.MULTIPLIER = 1.0  # only for mobilenet backbone

# ---------------------------------------------------------------------------- #
# PPM options
# ---------------------------------------------------------------------------- #
# supported backbone: resnet mobilenetv1 mobilenetv2
_C.PPM = CN()
_C.PPM.POOL_SCALES = (1, 2, 3, 6)
_C.PPM.PPM_HIDDEN_DIM = 512
_C.PPM.PPM_OUT_DIM = 512
_C.PPM.DROP_OUT = 0.1
_C.PPM.USE_AUX = True
_C.PPM.AUX_LOSS_WEIGHT = 0.4

# ---------------------------------------------------------------------------- #
# FCN options
# ---------------------------------------------------------------------------- #
# supported backbone: only vgg
_C.FCN = CN()
_C.FCN.DROP_OUT = 0.1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.ROOT = "../data/"

# Use GPU for training or testing if True
_C.GPU_USE = True

# Specify using GPU ids
_C.GPU_IDS = u'0,1,2,3,4,5,6,7'

# random seed
_C.SEED = 1024


def merge_cfg_from_file(file):
    if os.path.isfile(root_path() + file) and file.endswith('.yaml'):
        cfg.merge_from_file(root_path() + file)
    else:
        raise Exception('{} is not a yaml file'.format(file))


def merge_cfg_from_list(cfg_list):
    cfg.merge_from_list(cfg_list)


# for shown
def logger_cfg_from_file(file):
    return cfg.load_cfg(open(root_path() + file))

