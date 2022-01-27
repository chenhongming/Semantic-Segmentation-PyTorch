import os
import copy
import torch
import shutil
import argparse

from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms

import _init_path
from config.config import cfg, merge_cfg_from_file, merge_cfg_from_list, logger_cfg_from_file
from dataset import dataset, set_augmentations
from models.model_zone import generate_model, load_resume_state
from solver.loss import set_loss
from solver.optimizer import set_optimizer
from solver.scheduler import set_scheduler
from utils.utils import setup_logger, setup_seed, AverageMeter
from utils.plot import Writer
from utils.save import save_checkpoint
from utils.misc import check_mkdir, params_flops, get_lr, device_info
from utils.distributed import reduce_tensor, is_main_process
logger = setup_logger('debug-logger')


def demo():
    # Setup Config
    parser = argparse.ArgumentParser(description='Semantic Segmentation Model Debugging')
    parser.add_argument('--cfg', dest='cfg_file', default='../config/voc/voc_psp.yaml',
                        type=str, help='config file')
    parser.add_argument('opts', help='see ../config/config.py for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    logger.info("Called with args: {}".format(args))
    logger.info("Running with cfg:\n{}".format(logger_cfg_from_file(args.cfg_file)))

    # Setup Device
    is_distributed = False
    if cfg.GPU_USE:
        # Set temporary environment variables
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
            if torch.cuda.is_available():
                logger.info("Using Single GPU training!!!")
                logger.info("VISIBLE DEVICES (GPU) ID: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
                device = "cuda"
            else:
                device = 'cpu'
        else:
            if torch.cuda.is_available():
                is_distributed = True
                # init distributed training mode
                dist.init_process_group(backend="nccl", init_method="env://")
                torch.cuda.set_device(args.local_rank)
                dist.barrier()
                if is_main_process():
                    logger.info("Using Multi GPU training!!!")
                device = "cuda"
            else:
                device = 'cpu'
    else:
        logger.info("Using CPU training!!!")
        device = 'cpu'
    if is_main_process():
        device_info(device)

    # Setup Random Seed
    setup_seed(cfg.SEED)

    # Setup input_transform and augmentations
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
    ])
    input_augmentation = set_augmentations(cfg)

    # Setup Dataloader
    train_set = dataset.JsonDataset(json_path=cfg.DATA.TRAIN_JSON,
                                    split=cfg.MODEL.PHASE,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    crop_size=cfg.TRAIN.CROP_SIZE,
                                    padding=cfg.TRAIN.PADDING,
                                    ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                    transform=input_transform,
                                    augmentations=input_augmentation)
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                                               shuffle=(True if train_sampler is None else False),
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    model = generate_model().to(device)

    # Setup Optimizer
    optimizer = set_optimizer(model)

    # Setup Scheduler
    lr_scheduler = set_scheduler(optimizer)

    writer = Writer(cfg.CKPT)

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCH+1):
        for i, (image, target) in enumerate(train_loader):
            lr_scheduler.adjust_learning_rate(epoch, i, len(train_loader))
        lr = get_lr(optimizer)
        writer.append([epoch, lr])
    writer.draw_curve(cfg.MODEL.NAME)


def main():
    demo()


if __name__ == "__main__":
    main()
