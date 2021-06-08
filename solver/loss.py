import torch
import torch.nn as nn
from config.config import cfg


def set_loss():
    name = cfg.SOLVER.LOSS_NAME
    ignore_index = cfg.SOLVER.IGNORE_LABEL
    weight = torch.tensor(cfg.SOLVER.LOSS_WEIGHT, dtype=torch.float)

    if name == 'CrossEntropyLoss' or len(name) == 0:
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    else:
        raise Exception("Unsupported loss function: {}".format(name))
