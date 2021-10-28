import os
import time
import torch
import argparse
import cv2 as cv
from torchvision import transforms

import _init_path
from config.config import cfg, merge_cfg_from_file, merge_cfg_from_list, logger_cfg_from_file
from models.model_zone import generate_model
from models.backbone.build import load_trained_model
from utils.utils import setup_logger
from utils.color_map import set_colors
from utils.misc import check_mkdir, device_info
from utils.visualization import vis


def demo():
    # Setup Config
    parser = argparse.ArgumentParser(description='Semantic Segmentation Model Testing')
    parser.add_argument('--cfg', dest='cfg_file', default='../config/ade20k/ade20k_psp.yaml',
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
        logger.info("Using GPU testing!!!")
        logger.info("GPU ID: 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = "cuda"
    else:
        logger.info("Using CPU training!!!")
        device = 'cpu'
    device_info(device)

    # Setup input_transform and augmentations
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
    ])

    # Setup Model(load trained model weights)
    model = generate_model()
    model = load_trained_model(model, device)
    model.to(device).eval()

    # Setup result dir
    if not os.path.isdir(os.path.join(cfg.CKPT, 'results')):
        check_mkdir(os.path.join(cfg.CKPT, 'results'))

    # Setup Colors
    colors = set_colors(cfg.DATA.DATASET)

    # Setup Data loader, Forward, Visualize
    if cfg.TEST.MODE == 'image':
        files = os.listdir(cfg.TEST.IMAGE_PATH)
        for image_name in files:
            if image_name.endswith('.png') or image_name.endswith(".jpg"):
                image = cv.imread(cfg.TEST.IMAGE_PATH + image_name)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image_ = input_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(image_)
                pred = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()
                result = vis(pred, image, colors)
                logger.info("Testing image: {}".format(image_name))
                cv.imwrite(os.path.join(cfg.CKPT, 'results/') + image_name.replace(".", "_vis."), result)
            else:
                raise Exception("Incorrect image format: {}".format(image_name))
    elif cfg.TEST.MODE == 'video':
        # VIDEO_PATH ==    ""  :  realtime camera
        # VIDEO_PATH ==  "path":  local video
        if cfg.TEST.VIDEO_PATH == "":
            capture = cv.VideoCapture(0)
        else:
            capture = cv.VideoCapture(cfg.TEST.VIDEO_PATH)
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_ = input_transform(frame).unsqueeze(0).to(device)
            tic = time.time()
            with torch.no_grad():
                output = model(frame_)
            pred = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()
            result = vis(pred, frame, colors)
            fps = (1. / (time.time() - tic))
            result = cv.putText(result, "fps = %.2f" % fps, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow("result", result)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        capture.release()
        cv.destroyAllWindows()
    else:
        raise AssertionError("Please specify the correct mode: 'image','video'.")


def main():
    demo()


if __name__ == "__main__":
    main()
