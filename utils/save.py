import os
import time
import torch

from config.config import cfg
from utils.utils import setup_logger


def time_stamp():
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime())


def save_checkpoint(save_path, epoch, model, optimizer=None, lr_scheduler=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, '{}_{}_{}_Eopoch_{}_model.pth'.format(cfg.MODEL.NAME, cfg.MODEL.BACKBONE_NAME,
                                                                             cfg.DATA.DATASET, epoch))
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    save_state = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }

    logger = setup_logger('save-logger')
    torch.save(save_state, filename)
    logger.info('Epoch {} model saved in: {}'.format(epoch, filename))
