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

# supported cityscapes, ade20k, voc, voc_aug,
# camvid, kitti, mscoco, lip, mapillary
_C.DATA.DATASET = 'ade20k'
# index file of the data set
_C.DATA.TRAIN_JSON = ""
_C.DATA.VAL_JSON = ""
_C.DATA.TEST_JSON = ""
_C.DATA.CLASSES = 151
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------- #
# TRAIN options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
# BATCH_SIZE for per GPU
_C.TRAIN.BATCH_SIZE = 8
# Probability of using RandomFlip
_C.TRAIN.PROB = 0.5
# Ratio of using RandomResize
_C.TRAIN.RATIO = (0.5, 2)
# crop_size of using RandomCrop
# int: a square crop (crop_size, crop_size) is made
# tuple: (640, 512)
_C.TRAIN.CROP_SIZE = 512
_C.TRAIN.CROP_SIZE = (640, 512)
# params of using RandomRotate: PADDING for img, IGNORE_LABEL for mask
_C.TRAIN.ROTATE = (-10, 10)
_C.TRAIN.PADDING = (0, 0, 0)
_C.TRAIN.IGNORE_LABEL = 255

_C.TRAIN.START_EPOCH = 1
_C.TRAIN.MAX_EPOCH = 100
_C.TRAIN.SAVE_EPOCH = 2

# ---------------------------------------------------------------------------- #
# EVAL options
# ---------------------------------------------------------------------------- #
_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 2
_C.EVAL.CROP_SIZE = [640, 512]
# params of using RandomRotate: PADDING for img, IGNORE_LABEL for mask
_C.EVAL.PADDING = (0, 0, 0)
_C.EVAL.IGNORE_LABEL = 255

# ---------------------------------------------------------------------------- #
# EVAL options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# method: image or video
_C.TEST.MODE = 'image'
_C.TEST.IMAGE_PATH = ""
# "": realtime camera; "path": local video
_C.TEST.VIDEO_PATH = ""

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
# supported vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
# resnet18 resnet34 resnet50 resnet101 resnet152
# resnext50_32x4d resnext101_32x8d wide_resnet50_2 wide_resnet101_2
# densenet121 densenet161 densenet169 densenet201
# mobilenet_v1 mobilenet_v2 mobilenet_v3_small mobilenet_v3_large
# shufflenet_v1_g1 shufflenet_v1_g2 shufflenet_v1_g4 shufflenet_v1_g8
# shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0
# mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3
_C.MODEL.BACKBONE_NAME = 'resnet18'
_C.MODEL.BACKBONE_PRETRAINED = True
_C.MODEL.BACKBONE_WEIGHT = "pretrained/"
# bn or syncbn
# NOTE: The value(NORM_LAYER) does not need to be modified.
# The syncbn method is implemented by
#   model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
_C.MODEL.NORM_LAYER = 'bn'
# downsampling rate for image (i.e. 8x)
_C.MODEL.OUTPUT_STRIDE = 8  # 8, 16, 32
# only for resnet backbone
# This parameter determines whether to use 3 3X3 convolutions
# instead of 7X7 convolutions.
_C.MODEL.HEAD7X7 = False
# if test mode, RESUME and FINETUNE must be False
_C.MODEL.PHASE = 'train'
# load trained semantic segmentation model weight
_C.MODEL.TRAINED = False
# In the training process, if the training is interrupted,
# this parameter can continue to train without training from the scratch.
_C.MODEL.RESUME = False
# This parameter is used to finetune the segmentation model trained on
# the public semantic segmentation dataset to the actual priv dataset.
_C.MODEL.FINETUNE = False
# supported fcn32s fcn16s fcn8s deeplabv3 deeplabv3plus
# lraspp denseaspp psp bisenet contextnet
_C.MODEL.NAME = 'psp'
# Used to store the segmentation model path after training
_C.MODEL.MODEL_WEIGHT = ""
# for mobilenetv1-v2 shufflenetv1-v2 backbone
_C.MODEL.MULTIPLIER = 1.0
# for densenet
_C.MODEL.DROP_RATE = 0.1
# c3 must be not None if USE_AUX is True. See segmentation model define
_C.MODEL.USE_AUX = True
_C.MODEL.AUX_LOSS_WEIGHT = 0.4
# for bisenet
_C.MODEL.AUX2_LOSS_WEIGHT = 0.4

# ---------------------------------------------------------------------------- #
# PSP(PPM) options
# ---------------------------------------------------------------------------- #
# supported backbone: resnet mobilenetv1 mobilenetv2 shufflenetv2
_C.PPM = CN()
_C.PPM.POOL_SCALES = (1, 2, 3, 6)
_C.PPM.PPM_HIDDEN_DIM = 512
_C.PPM.PPM_OUT_DIM = 512
_C.PPM.DROP_OUT = 0.1

# ---------------------------------------------------------------------------- #
# FCN options
# ---------------------------------------------------------------------------- #
# supported backbone: only vgg
_C.FCN = CN()
_C.FCN.DROP_OUT = 0.1

# ---------------------------------------------------------------------------- #
# deeplabv3(ASPP) options
# ---------------------------------------------------------------------------- #
# supported backbone: resnet mobilenetv1 mobilenetv2 shufflenetv2
_C.ASPP = CN()
_C.ASPP.OUTPUT_STRIDE = 8  # 8 or 16
_C.ASPP.OUT_CHANNELS = 512
_C.ASPP.DROPOUT = 0.5

# ---------------------------------------------------------------------------- #
# deeplabv3plus options
# ---------------------------------------------------------------------------- #
_C.DEEPLABV3PLUS = CN()
_C.DEEPLABV3PLUS.LOW_LEVEL_FEATURE_CHANNELS = 48  # 48 or 32
# only 16 for deeplabv3plus
_C.DEEPLABV3PLUS.OUTPUT_STRIDE = 16
_C.DEEPLABV3PLUS.DROPOUT = 0.1

# ---------------------------------------------------------------------------- #
# lraspp options
# ---------------------------------------------------------------------------- #
_C.LRASPP = CN()
_C.LRASPP.INTER_CHANNELS = 128  # default

# ---------------------------------------------------------------------------- #
# denseaspp options
# ---------------------------------------------------------------------------- #
_C.DENSEASPP = CN()
_C.DENSEASPP.OUTPUT_STRIDE = 8
_C.DENSEASPP.INTER_CHANNELS = 256  # default
_C.DENSEASPP.OUT_CHANNELS = 64  # default
_C.DENSEASPP.ATROUS_RATE = [3, 6, 12, 18, 24]  # default
_C.DENSEASPP.DROP_RATE = 0.1

# ---------------------------------------------------------------------------- #
# bisenet options
# ---------------------------------------------------------------------------- #
_C.BISENET = CN()
_C.BISENET.IN_CHANNELS = 3
_C.BISENET.SPATIAL_PATH_OUT_CHANNELS = 128
_C.BISENET.CONTEXT_PATH_OUT_CHANNELS = 128
_C.BISENET.DROP_RATE = 0.1

# ---------------------------------------------------------------------------- #
# contextnet options
# ---------------------------------------------------------------------------- #
_C.ContextNet = CN()
_C.ContextNet.IN_CHANNELS = 3
_C.ContextNet.CHANNELS = [32, 32, 48, 64, 96, 128]
_C.ContextNet.EXPANSION_FACTOR = [1, 6, 6, 6, 6, 6]
_C.ContextNet.LAYERS = [1, 1, 3, 3, 2, 2]
_C.ContextNet.HIDDEN_CHANNELS1 = 32
_C.ContextNet.HIDDEN_CHANNELS2 = 64
_C.ContextNet.OUT_CHANNELS = 128

# ---------------------------------------------------------------------------- #
# SOLVER options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.LOSS_NAME = ''  # only CrossEntropyLoss
# Used to solve the problem of category imbalance
_C.SOLVER.LOSS_WEIGHT = []
# Category not used to calculate loss
_C.SOLVER.IGNORE_LABEL = 255
# sgd adam asgd adamax adadelta adagrad rmsprop
_C.SOLVER.OPTIMIZER_NAME = 'sgd'
_C.SOLVER.LR = 0.02
_C.SOLVER.MOMENTUM = 0.99
_C.SOLVER.WEIGHT_DECAY = 4e-5
# Adadelta: 1e-6; Adam,Adamax,RMSprop: 1e-8; Adagrad: 1e-10
_C.SOLVER.EPS = 1e-8
_C.SOLVER.BETAS = (0.9, 0.999)  # for Adam, Adamax
_C.SOLVER.AMSGRAD = False  # for Adam
_C.SOLVER.LAMBD = 1e-4  # for ASGD
_C.SOLVER.ALPHA = 0.75  # for ASGD:0.75; RMSprop:0.99
_C.SOLVER.T0 = 1e6  # for ASGD
_C.SOLVER.RTO = 0.9  # for Adadelta
_C.SOLVER.LR_DECAY = 0  # for Adagrad

# 'step', 'multistep', 'exponential', 'CosineAnnealing'
# (lr update by epoch method)
_C.SOLVER.SCHEDULER_NAME = 'step'
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEP_SIZE = 10  # for 'step'
_C.SOLVER.MILESTONES = [30, 60, 90]  # for 'multistep'
_C.SOLVER.T_MAX = 100  # for 'CosineAnnealing'

# another method: 'poly', 'step', 'cosine'
# (lr update by batch size method)
_C.SOLVER.LR_POLICY = 'poly'
# Warm up to SOLVER.LR over this number of sgd epochs
_C.SOLVER.WARM_UP_EPOCH = 0
# LR of warm up beginning
_C.SOLVER.WARM_UP_LR = 0.002
# For 'ploy', the power in poly to drop LR
_C.SOLVER.LR_POW = 0.9
# For 'STEP', Non-uniform step iterations
_C.SOLVER.STEPS = [30, 60, 90]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.ROOT = "../data/"

# Use GPU for training or testing if True
_C.GPU_USE = True

# Specify using GPU id for Single GPU training
_C.GPU_ID = u'0'

# random seed
_C.SEED = 1024

# Directory for saving checkpoints and loggers
_C.CKPT = './ckpts/ade20k/ade20k_psp'


def merge_cfg_from_file(file):
    if os.path.isfile(file) and file.endswith('.yaml'):
        cfg.merge_from_file(file)
    else:
        raise Exception('{} is not a yaml file'.format(file))


def merge_cfg_from_list(cfg_list):
    cfg.merge_from_list(cfg_list)


# for shown
def logger_cfg_from_file(file):
    return cfg.load_cfg(open(file))

