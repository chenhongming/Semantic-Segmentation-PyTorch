import os
import time
import torch
import shutil

from config.config import cfg
from utils.utils import setup_logger
logger = setup_logger('save-logger')


def time_stamp():
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime())


def save_checkpoint(save_path, epoch, model, optimizer=None, is_best=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = '{}_{}_{}_model.pth'.format(cfg.MODEL.NAME, cfg.MODEL.BACKBONE_NAME, cfg.DATA.DATASET)
    filename = os.path.join(save_path, filename)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    save_state = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
    }

    torch.save(save_state, filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(cfg.MODEL.NAME, cfg.MODEL.BACKBONE_NAME, cfg.DATA.DATASET)
        best_filename = os.path.join(save_path, best_filename)
        shutil.copyfile(filename, best_filename)
        logger.info('best model saved in: {}'.format(best_filename))
