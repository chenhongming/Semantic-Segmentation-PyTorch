import os
import torch

from config.config import cfg
from utils.registry import Registry
from utils.utils import setup_logger, root_path
from utils.distributed import is_main_process


MODEL_REGISTRY = Registry('model')
logger = setup_logger('model_zone-logger')


# generate a semantic segmentation model using cfg prams
def generate_model():
    name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(name)()
    if cfg.MODEL.TRAINED:
        model = load_trained_model(model)
    return model


# load trained semantic segmentation model weight
def load_trained_model(model):
    if cfg.MODEL.PHASE == 'train' and not cfg.MODEL.RESUME and not cfg.MODEL.FINETUNE:
        if is_main_process() and cfg.MODEL.BACKBONE_NAME is None:
            logger.info("{} use kaiming_normal init...".format(cfg.MODEL.NAME))
        return model
    else:
        model_file = root_path() + cfg.MODEL.MODEL_WEIGHT
        suffix = os.path.splitext(model_file)[-1]
        if os.path.isfile(model_file) and suffix == '.pth':
            model.load_state_dict(torch.load(model_file))
            if is_main_process() and cfg.MODEL.RESUME:
                logger.info("Resume training, loading {}... ".format(model_file))
            elif is_main_process() and cfg.MODEL.FINETUNE:
                logger.info("Finetune training, loading {}... ".format(model_file))
            else:
                if is_main_process():
                    logger.info("Test, loading {}... ".format(model_file))
        else:
            if is_main_process():
                raise Exception("{} is not a valid pth file".format(model_file))
        return model


def load_resume_state():
    state_file = root_path() + cfg.MODEL.MODEL_WEIGHT
    suffix = os.path.splitext(state_file)[-1]
    if os.path.isfile(state_file) and suffix == '.pth':
        return torch.load(state_file)
    else:
        if is_main_process():
            raise Exception("{} is not a valid pth file".format(state_file))

