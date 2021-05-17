import os
from yacs.config import CfgNode as CN

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
_C.DATA.CLASSES = 0
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------- #
# TRAIN options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.ARCH = ""
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
# PPM options
# ---------------------------------------------------------------------------- #
_C.PPM = CN()
_C.PPM.POOL_SCALES = (1, 2, 3, 6)

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


def load_defaults_cfg():
    """
    Get a yacs CfgNode object with default values.
    Return a clone so that the defaults will not be altered
    """
    return _C.clone()


def merge_cfg_from_file(cfg, file):
    if os.path.isfile(file) and file.endswith('.yaml'):
        cfg.merge_from_file(file)
        cfg.freeze()
    else:
        raise Exception('{} is not a yaml file'.format(file))


def merge_cfg_from_list(cfg, cfg_list):
    cfg.merge_from_list(cfg_list)


def logger_cfg_from_file(file):
    return CN.load_cfg(open(file))

