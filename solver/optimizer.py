from config.config import cfg
from torch import optim


def set_optimizer(model):
    param_groups = model.parameters()
    name = cfg.SOLVER.OPTIMIZER_NAME.lower()

    if name == 'sgd':
        optimizer = optim.SGD(param_groups, lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM,
                              weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif name == 'adam':
        optimizer = optim.Adam(param_groups, lr=cfg.SOLVER.LR, betas=cfg.SOLVER.BETAS, eps=cfg.SOLVER.EPS,
                               weight_decay=cfg.SOLVER.WEIGHT_DECAY, amsgrad=cfg.SOLVER.AMSGRAD)
    elif name == 'asgd':
        optimizer = optim.ASGD(param_groups, lr=cfg.SOLVER.LR, lambd=cfg.SOLVER.LAMBD, alpha=cfg.SOLVER.ALPHA,
                               t0=cfg.SOLVER.T0, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif name == 'adamax':
        optimizer = optim.Adamax(param_groups, lr=cfg.SOLVER.LR, betas=cfg.SOLVER.BETAS, eps=cfg.SOLVER.EPS,
                                 weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif name == 'adadelta':
        optimizer = optim.Adadelta(param_groups, lr=cfg.SOLVER.LR, rho=cfg.SOLVER.RTO, eps=cfg.SOLVER.EPS,
                                   weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif name == 'adagrad':
        optimizer = optim.Adagrad(param_groups, lr=cfg.SOLVER.LR, lr_decay=cfg.SOLVER.LR_DECAY,
                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY, eps=cfg.SOLVER.EPS)
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(param_groups, lr=cfg.SOLVER.LR, alpha=cfg.SOLVER.ALPHA, eps=cfg.SOLVER.EPS,
                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
    return optimizer
