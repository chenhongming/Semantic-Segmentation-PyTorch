import os
import torch
import argparse

from torchvision import transforms

import _init_path
from config import config
from utils.utils import setup_logger, setup_seed
from dataset import dataset, set_augmentations
from models.pspnet import PSPNet


def main():
    # Setup Config
    parser = argparse.ArgumentParser(description='Semantic Segmentation Model Training')
    parser.add_argument('--cfg', dest='cfg_file', default='../config/ade20k/ade20k_resnet18_ppm.yaml',
                        type=str, help='config file')
    parser.add_argument('opts', help='see ../config/config.py for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = config.load_defaults_cfg()  # all config options
    if args.cfg_file is not None:
        config.merge_cfg_from_file(cfg, args.cfg_file)
    if args.opts is not None:
        config.merge_cfg_from_list(cfg, args.opts)
    logger = setup_logger('main-logger')
    logger.info("Called with args: {}".format(args))
    logger.info("Running with cfg:\n{}".format(config.logger_cfg_from_file(args.cfg_file)))

    # Setup Device
    if torch.cuda.is_available() and cfg.GPU_USE:
        # Set temporary environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.GPU_IDS)
        device = "cuda"
    else:
        device = 'cpu'

    # Setup Random Seed
    setup_seed(cfg.SEED)

    # Setup input_transform and augmentations
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
    ])
    input_augmentation = set_augmentations(cfg)
    # Setup Dataloader
    val_set = dataset.JsonDataset(json_path=cfg.DATA.VAL_JSON, transform=input_transform,
                                  augmentations=input_augmentation)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=None, pin_memory=True, sampler=None, drop_last=True)
    # for i, (inputs, target) in enumerate(val_loader):
    #     print(i)
    # Setup Model
    model = PSPNet()
    print(model)


if __name__ == '__main__':
    main()
