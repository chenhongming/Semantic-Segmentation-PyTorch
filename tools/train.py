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
from utils.metrics import accuracy, intersectionAndUnion
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
            model.load_state_dict(ckpt_state['state_dict'])
            optimizer.load_state_dict(ckpt_state['optimizer'])

            cfg.TRAIN.START_EPOCH = ckpt_state['epoch']
            cfg.SOLVER.LR = [param_group['lr'] for param_group in optimizer.param_groups][0]
            logger.info('resume train from epoch: {}'.format(cfg.TRAIN.START_EPOCH))
            logger.info('resume optimizer and lr from resume state...')

    # Setup Output dir
    if is_main_process():
        if not os.path.isdir(cfg.CKPT):
            check_mkdir(cfg.CKPT)
        if args.cfg_file is not None:
            shutil.copyfile(args.cfg_file, os.path.join(cfg.CKPT, args.cfg_file.split('/')[-1]))

    # Setup Draw curve
    writer = Writer(cfg.CKPT)

    # main loop
    is_best = False
    if is_main_process():
        logger.info("\n\t\t\t>>>>> Start training >>>>>")
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCH+1):
        train_loss = train(model, train_loader, criterion, optimizer, lr_scheduler, epoch, device, is_distributed, writer)
        if not cfg.TRAIN.SKIP_VAL and cfg.TRAIN.VAL_EPOCH_INTERVAL:
            is_best = validation(model, train_loader, device, is_distributed, is_best)
        if is_distributed:
            train_sampler.set_epoch(epoch)
        if is_main_process():
            writer.append([epoch, train_loss])
            writer.draw_curve(cfg.MODEL.NAME)
            if epoch % cfg.TRAIN.SAVE_EPOCH_INTERVAL == 0:
                save_checkpoint(cfg.CKPT, epoch, model, optimizer, is_best)


def train(model, loader, criterion, optimizer, lr_scheduler, epoch, device, is_distributed, writer):
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
            lr_scheduler.adjust_learning_rate(epoch, index, len(loader))

            # record loss
            ave_total_loss.update(loss.item())
            if is_main_process() and epoch == 1 and index == 0:
                writer.append([epoch-1, ave_total_loss.avg])

            # display msg
            pbar.set_postfix(**{'loss': ave_total_loss.avg, 'lr': get_lr(optimizer)})
            pbar.update(1)
    return ave_total_loss.avg


def validation(model, loader, device, is_distributed, is_best):
    if is_distributed:
        model = model.module
    torch.cuda.empty_cache()

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    best_pred_meter = AverageMeter()

    # switch to eval model
    model.to(device).eval()
    desc = 'Validation'
    with tqdm(total=len(loader), desc=desc, leave=False) as pbar:
        for index, (images, masks) in enumerate(loader):
            # load data to device
            images = images.to(device)
            labels = masks.numpy()

            with torch.no_grad():
                # forward
                outputs = model(images)
                preds = outputs.data.max(1)[1].cpu().numpy()

            acc, pix = accuracy(preds, labels)
            intersection, union = intersectionAndUnion(preds, labels, cfg.DATA.CLASSES, cfg.EVAL.IGNORE_LABEL)

            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)
            iou = (intersection_meter.sum / (union_meter.sum + 1e-10)).mean()
            # display msg
            pbar.set_postfix(**{'Acc': acc_meter.avg.item() * 100, 'mIoU': iou})
            pbar.update(1)
    new_pred = (acc_meter.avg.item() + iou) / 2
    if new_pred > best_pred_meter.val:
        is_best = True
        best_pred_meter.update(new_pred)
    model.train()
    # logger.info("best_pred_meter.val: {}".format(best_pred_meter.val))
    return is_best


if __name__ == '__main__':
    main()
