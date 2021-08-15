import torch
import torch.nn as nn
from config.config import cfg


def set_loss():
    return Criterion()


class Criterion(nn.Module):

    def __init__(self):
        super().__init__()
        name = cfg.SOLVER.LOSS_NAME
        ignore_index = cfg.SOLVER.IGNORE_LABEL
        weight = torch.tensor(cfg.SOLVER.LOSS_WEIGHT, dtype=torch.float)

        if name == 'CrossEntropyLoss' or len(name) == 0:
            if len(weight) == cfg.DATA.CLASSES:
                self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            else:
                self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            raise Exception("Unsupported loss function: {}".format(name))

    def forward(self, preds, labels):
        if len(preds) == 2:
            print(preds[0].shape, labels.shape)
            main_loss = self.criterion(preds[0], labels)
            aux_loss = self.criterion(preds[1], labels)
            loss = main_loss + cfg.MODEL.AUX_LOSS_WEIGHT * aux_loss
            return loss
        elif len(preds) == 3:  # for 'bisenet'
            main_loss = self.criterion(preds[0], labels)
            aux1_loss = self.criterion(preds[1], labels)
            aux2_loss = self.criterion(preds[2], labels)
            loss = main_loss + cfg.MODEL.AUX_LOSS_WEIGHT * aux1_loss + cfg.MODEL.AUX2_LOSS_WEIGHT * aux2_loss
            return loss
        print(preds[0].shape, labels.shape)
        loss = self.criterion(preds, labels)
        return loss
