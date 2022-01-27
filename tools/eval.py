import os
import time
import torch
import argparse

from torchvision import transforms

import _init_path
from config.config import cfg, merge_cfg_from_file, merge_cfg_from_list, logger_cfg_from_file
from dataset import dataset
from models.backbone.build import load_trained_model
from models.model_zone import generate_model
from utils.misc import device_info
from utils.utils import setup_logger, AverageMeter
from utils.metrics import accuracy, intersectionAndUnion


def main():
    # Setup Config
    parser = argparse.ArgumentParser(description='Semantic Segmentation Model Evaluating')
    parser.add_argument('--cfg', dest='cfg_file', default='../config/voc/voc_fcn32s.yaml',
                        type=str, help='config file')
    parser.add_argument('opts', help='see ../config/config.py for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    logger = setup_logger('main-logger')
    logger.info("Called with args: {}".format(args))
    logger.info("Running with cfg:\n{}".format(logger_cfg_from_file(args.cfg_file)))

    # Setup Device
    if torch.cuda.is_available() and cfg.GPU_USE:
        # Set temporary environment variables
        logger.info("Using GPU evaluating!!!")
        logger.info("GPU ID: 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = "cuda"
    else:
        logger.info("Using CPU evaluating!!!")
        device = 'cpu'
    device_info(device)

    # Setup input_transform and augmentations
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
    ])

    # Setup Dataloader
    eval_set = dataset.JsonDataset(json_path=cfg.DATA.VAL_JSON,
                                   dataset=cfg.DATA.DATASET,
                                   batch_size=cfg.VAL.BATCH_SIZE,
                                   crop_size=cfg.VAL.CROP_SIZE,
                                   padding=cfg.VAL.PADDING,
                                   ignore_label=cfg.VAL.IGNORE_LABEL,
                                   transform=input_transform)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=cfg.VAL.BATCH_SIZE, shuffle=None,
                                               pin_memory=True, sampler=None, drop_last=False)

    # Setup Model
    model = generate_model()
    logger.info("Evaluating model:\n\033[1;34m{} \033[0m".format(model))

    # load trained model weights
    model = load_trained_model(model, device)

    # main loop
    logger.info("\n\t\t\t>>>>> Start Evaluating >>>>>")
    eval(model, eval_loader, device, logger)


def eval(model, loader, device, logger):
    # switch to eval model
    model.to(device).eval()

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    data_time = AverageMeter()
    infer_time = AverageMeter()

    num_images = len(loader) * cfg.VAL.BATCH_SIZE
    tic = time.time()
    logger.info('num_images: {} | batch_size: {}'.format(num_images, cfg.VAL.BATCH_SIZE))
    for index, (images, masks) in enumerate(loader):
        # load data to device
        images = images.to(device)
        labels = masks.numpy()

        # measure data loading time
        data_time.update((time.time() - tic) / cfg.VAL.BATCH_SIZE)
        tic_ = time.time()

        with torch.no_grad():
            # forward
            outputs = model(images)
            preds = outputs.data.max(1)[1].cpu().numpy()

        infer_time.update((time.time() - tic_) / cfg.VAL.BATCH_SIZE)

        acc, pix = accuracy(preds, labels)
        intersection, union = intersectionAndUnion(preds, labels, cfg.DATA.CLASSES, cfg.VAL.IGNORE_LABEL)

        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        tic = time.time()
        logger.info('Evaluating: iter:{}/{}\n | mean_acc: {:4.4f} % | cur_acc: {:4.4f} % | infer_time_avg: {:.3f} s'
                    .format(index, len(loader), acc_meter.avg * 100., acc * 100., infer_time.avg))
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        logger.info('class [{}], IoU: {:.4}'.format(i+1, _iou))
    logger.info('[Eval Summary]: Mean IoU: {:.4}, Accuracy: {:.2f}%'.format(iou.mean(), acc_meter.avg * 100.))


if __name__ == '__main__':
    main()
