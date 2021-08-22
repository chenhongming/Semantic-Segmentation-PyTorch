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
    if cfg.MODEL.BACKBONE_PRETRAINED:
        model = load_pretrained_model(model)
    return model


# load pretrained semantic segmentation model weight
def load_pretrained_model(model):
    logger = setup_logger('model_zone-logger')
    if cfg.MODEL.PHASE == 'train' and not cfg.MODEL.RESUME and not cfg.MODEL.FINETUNE:
        if cfg.MODEL.BACKBONE_NAME is None:
            logger.info("{} use kaiming_normal init...".format(cfg.MODEL.NAME))
        return model
    else:
        model_file = root_path() + cfg.MODEL.MODEL_WEIGHT
        suffix = os.path.splitext(model_file)[-1]
        if os.path.isfile(model_file) and suffix == '.pth':
            model.load_state_dict(torch.load(model_file))
            if cfg.MODEL.RESUME:
                logger.info("Resume training, loading {}... ".format(model_file))
            elif cfg.MODEL.FINETUNE:
                logger.info("Finetune training, loading {}... ".format(model_file))
            else:
                logger.info("Test, loading {}... ".format(model_file))
        else:
            raise Exception("{} is not a valid pth file".format(model_file))
        return model


def load_resume_state():
    state_file = root_path() + cfg.MODEL.MODEL_WEIGHT
    suffix = os.path.splitext(state_file)[-1]
    if os.path.isfile(state_file) and suffix == '.pth':
        return torch.load(state_file)
    else:
        raise Exception("{} is not a valid pth file".format(state_file))

