import os
import torch
from utils.registry import Registry
from utils.utils import setup_logger,root_path
from config.config import cfg

MODEL_REGISTRY = Registry('model')


# generate a semantic segmentation model using cfg prams
def generate_model():
    name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(name)()
    if cfg.MODEL.PRETRAINED:
        model = load_pretrained_model(model)
    return model


# load pretrained semantic segmentation model weight
def load_pretrained_model(model):
    if cfg.MODEL.PHASE == 'train' and not cfg.MODEL.RESUME and not cfg.MODEL.FINETUNE:
        if cfg.MODEL.BACKBONE_NAME is None:
            logger = setup_logger('model_zone-logger')
            logger.info("{} use kaiming_normal init...".format(cfg.MODEL.NAME))
        return model
    else:
        pretrained_file = root_path() + cfg.MODEL.MODEL_WEIGHT
        suffix = os.path.splitext(pretrained_file)[-1]
        if os.path.isfile(pretrained_file) and suffix == '.pth':
            model.load_state_dict(torch.load(pretrained_file))
        else:
            raise Exception("{} is not a valid pth file".format(pretrained_file))
