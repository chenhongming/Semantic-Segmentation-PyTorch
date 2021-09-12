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
logger = setup_logger('main-logger')


def main():
    # Setup Config
    parser = argparse.ArgumentParser(description='Semantic Segmentation Model Training')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/ade20k/ade20k_psp.yaml',
                        type=str, help='config file')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('opts', help='see ./config/config.py for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    logger.info("Called with args: {}".format(args))
    # logger.info("Running with cfg:\n{}".format(logger_cfg_from_file(args.cfg_file)))

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

    # Setup Model
    model = generate_model().to(device)
    if is_main_process():
        logger.info("Training model:\n\033[1;34m{} \033[0m".format(model))

    # Setup Params and Flops
    if is_main_process():
        params_flops(copy.deepcopy(model), cfg.TRAIN.CROP_SIZE, device)

    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Setup Loss
    criterion = set_loss().to(device)

    # Setup Optimizer
    optimizer = set_optimizer(model)

    # Setup Scheduler
    lr_scheduler = set_scheduler(optimizer)

    # Setup Resume
    if cfg.MODEL.RESUME:
        if is_main_process():
            ckpt_state = load_resume_state()
            cfg.TRAIN.START_EPOCH = ckpt_state['epoch']
            logger.info('resume train from epoch: {}'.format(cfg.TRAIN.START_EPOCH))
            if ckpt_state['optimizer'] is not None and ckpt_state['lr_scheduler'] is not None:
                optimizer.load_state_dict(ckpt_state['optimizer'])
                lr_scheduler.load_state_dict(ckpt_state['lr_scheduler'])
                logger.info('resume optimizer and lr scheduler from resume state...')

    # Setup Output dir
    if is_main_process():
        if not os.path.isdir(cfg.CKPT):
            check_mkdir(cfg.CKPT)
        if args.cfg_file is not None:
            shutil.copyfile(args.cfg_file, os.path.join(cfg.CKPT, args.cfg_file.split('/')[-1]))

    # Setup Draw curve
    writer = Writer(cfg.CKPT)

    # main loop
    if is_main_process():
        logger.info("\n\t\t\t>>>>> Start training >>>>>")
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCH+1):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, device, is_distributed)
        lr_scheduler.step()
        if is_distributed:
            train_sampler.set_epoch(epoch)
        if is_main_process():
            writer.append([epoch, train_loss])
            writer.draw_curve(cfg.MODEL.NAME)
            if epoch % cfg.TRAIN.SAVE_EPOCH == 0:
                save_checkpoint(cfg.CKPT, epoch, model, optimizer, lr_scheduler)


def train(model, loader, criterion, optimizer, epoch, device, is_distributed):
    ave_total_loss = AverageMeter()
    # switch to train model
    model.train()
    desc = f'Epoch {epoch}/{cfg.TRAIN.MAX_EPOCH}'
    with tqdm(total=len(loader), desc=desc, leave=False) as pbar:
        for index, (images, masks) in enumerate(loader):
            # load data to device
            images = images.to(device)
            masks = masks.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss.mean()
            if is_distributed:
                loss = reduce_tensor(loss)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            ave_total_loss.update(loss.item())

            # display msg
            pbar.set_postfix(**{'loss': ave_total_loss.avg, 'lr': get_lr(optimizer)})
            pbar.update(1)
    return ave_total_loss.avg


if __name__ == '__main__':
    main()
