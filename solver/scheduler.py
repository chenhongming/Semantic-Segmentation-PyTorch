import numpy as np
from torch.optim import lr_scheduler

from config.config import cfg


# The lr_scheduler updated every epoch by calling scheduler.step()
# for epoch in cfg.SOLVER.MAX_EPOCHS:
#   train()
#   scheduler.step()
def set_scheduler(optimizer):
    name = cfg.SOLVER.SCHEDULER_NAME.lower()

    if name == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.STEP_SIZE, gamma=cfg.SOLVER.GAMMA)
    elif name == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=cfg.SOLVER.GAMMA)
    elif name == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)
    elif name == 'CosineAnnealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.T_MAX)
    else:
        raise Exception("Unsupported scheduler: {}".format(name))
    return scheduler


# The lr_scheduler updated every batch size
def set_scheduler1(optimizer):
    scheduler = Scheduler(optimizer)
    return scheduler


class Scheduler:
    cur_lr = cfg.SOLVER.LR

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch=1, iteration=1, iter_per_epoch=1000):
        assert cfg.SOLVER.LR_POLICY.lower() in ['step', 'cosine', 'poly']

        if (epoch - 1) < cfg.SOLVER.WARM_UP_EPOCH:
            cur_iter = ((epoch - 1) % cfg.SOLVER.WARM_UP_EPOCH) * iter_per_epoch + iteration
            lr_new = cfg.SOLVER.WARM_UP_LR + (cfg.SOLVER.LR - cfg.SOLVER.WARM_UP_LR) * cur_iter / (
                        iter_per_epoch * cfg.SOLVER.WARM_UP_EPOCH)
        else:
            if cfg.SOLVER.LR_POLICY == 'step':
                if epoch in cfg.SOLVER.STEPS and iteration == 1:
                    self.cur_lr *= cfg.SOLVER.GAMMA ** (cfg.SOLVER.STEPS.index(epoch) + 1)
                lr_new = self.cur_lr
            elif cfg.SOLVER.LR_POLICY == 'cosine':  # except warm up
                total_iter = (cfg.SOLVER.MAX_EPOCHS - cfg.SOLVER.WARM_UP_EPOCH) * iter_per_epoch
                cur_iter = ((epoch - cfg.SOLVER.WARM_UP_EPOCH - 1) % cfg.SOLVER.MAX_EPOCHS) * iter_per_epoch + iteration
                lr_new = 0.5 * cfg.SOLVER.LR * (np.cos(cur_iter * np.pi / total_iter) + 1.0)
            elif cfg.SOLVER.LR_POLICY == 'poly':  # except warm up
                total_iter = (cfg.SOLVER.MAX_EPOCHS - cfg.SOLVER.WARM_UP_EPOCH) * iter_per_epoch
                cur_iter = ((epoch - cfg.SOLVER.WARM_UP_EPOCH - 1) % cfg.SOLVER.MAX_EPOCHS) * iter_per_epoch + iteration
                scale_lr = ((1. - float(cur_iter) / total_iter) ** cfg.SOLVER.LR_POW)
                lr_new = scale_lr * cfg.SOLVER.LR
            else:
                raise KeyError('Unknown SOLVER.LR_POLICY: {}'.format(cfg.SOLVER.LR_POLICY))

        # assign new lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_new
