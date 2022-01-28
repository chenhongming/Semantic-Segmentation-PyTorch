from PIL import Image
import numpy as np
import cv2 as cv
import random
import os
import json
from PIL import ImageOps
import torchvision.transforms.functional as F

import torch
from torchviz import make_dot

"""
root = "/Users/chenhongming/Desktop/segmentation/Semantic_Segmentation_PyTorch/data/"

color_map = {0: [0, 0, 0], 1: [0, 0, 128], 2: [0, 128, 0], 3: [0, 128, 128], 4: [128, 0, 0], 5: [128, 0, 128],
         6: [128, 128, 0], 7: [128, 128, 128], 8: [0, 0, 64], 9: [0, 0, 192], 10: [0, 128, 64],
         11: [0, 128, 192], 12: [128, 0, 64], 13: [128, 0, 192], 14: [128, 128, 64], 15: [128, 128, 192],
         16: [0, 64, 0], 17: [0, 64, 128], 18: [0, 192, 0], 19: [0, 192, 128], 20: [128, 64, 0],
         255: [255, 255, 255]}


def load_sample(json_name):
    with open(root + json_name, 'r') as f:
        json_file = json.load(f)
    return json_file


def vis(image, mask):
    value = np.unique(mask)
    print('mask value: ', value)
    # print("image [100][100]: ", image.getpixel((100, 100)))
    # w, h = mask.size
    # mask = np.asarray(mask)
    # out = np.zeros((h, w, 3)).astype('uint8')
    # for i in value:
    #     out[np.where(mask == i)] = color_map[i]
    # # cv.imwrite("vis.png", out)
    # cv.imshow('out', out)
    # cv.waitKey(0)


class RandomRotate:
    def __init__(self, rotate=(-10, 10), padding=(0, 0, 0), ignore_label=255):
        self.rotate = rotate
        self.padding = padding
        self.ignore_label = ignore_label

    def __call__(self, img, mask):
        assert img.size == mask.size
        assert len(self.rotate) == 2 and len(self.padding) == 3
        rotate_degree = random.randint(self.rotate[0], self.rotate[1])
        img = F.affine(img, angle=rotate_degree, interpolation=F.InterpolationMode.BILINEAR, fill=self.padding,
                       translate=(0, 0), scale=1.0, shear=0.0)
        mask = F.affine(mask, angle=rotate_degree, interpolation=F.InterpolationMode.NEAREST, fill=self.ignore_label,
                        translate=(0, 0), scale=1.0, shear=0.0)
        return img, mask


sample_list = load_sample("Pascal_VOC/train.json")
r = RandomRotate()
for i in range(len(sample_list)):
    image = Image.open(root + sample_list[i].split(' ')[0])
    mask = Image.open(root + sample_list[i].split(' ')[1])
    print("name: ", sample_list[i].split(' ')[1])
    print("before r:")
    vis(image, mask)
    t_image, t_mask = r(image, mask)
    print("after r:")
    vis(t_image, t_mask)
    print("===============")
"""


# print(mask.size)
color_map = {0: [0, 0, 0], 1: [0, 0, 128], 2: [0, 128, 0], 3: [0, 128, 128], 4: [128, 0, 0], 5: [128, 0, 128],
         6: [128, 128, 0], 7: [128, 128, 128], 8: [0, 0, 64], 9: [0, 0, 192], 10: [0, 128, 64],
         11: [0, 128, 192], 12: [128, 0, 64], 13: [128, 0, 192], 14: [128, 128, 64], 15: [128, 128, 192],
         16: [0, 64, 0], 17: [0, 64, 128], 18: [0, 192, 0], 19: [0, 192, 128], 20: [128, 64, 0],
         255: [255, 255, 255]}

# w, h = mask.size
# mask = np.asarray(mask)
# out = np.zeros((h, w, 3)).astype('uint8')
# for i in value:
#     out[np.where(mask == i)] = color_map[i]
# cv.imwrite("vis.png", out)
# cv.imshow('out', out)
# cv.waitKey(0)


class RandomRotate:
    def __init__(self, rotate=(0, 0), padding=(0, 0, 0), ignore_label=255):
        self.rotate = rotate
        self.padding = padding
        self.ignore_label = ignore_label

    def __call__(self, img, mask):
        assert img.size == mask.size
        assert len(self.rotate) == 2 and len(self.padding) == 3
        rotate_degree = random.randint(self.rotate[0], self.rotate[1])
        if rotate_degree != 0:
            img = F.affine(img, angle=rotate_degree, interpolation=F.InterpolationMode.BILINEAR, fill=self.padding,
                           translate=(0, 0), scale=1.0, shear=0.0)
            mask = F.affine(mask, angle=rotate_degree, interpolation=F.InterpolationMode.NEAREST, fill=self.ignore_label,
                            translate=(0, 0), scale=1.0, shear=0.0)
        # img.save("/Users/chenhongming/Desktop/image/001311.jpg")
        # mask.save("/Users/chenhongming/Desktop/image/001311.png")
        return img, mask


def unified_size(image, mask, crop_size, padding=(0, 0, 0), ignore_label=255):
    w, h = image.size
    if isinstance(crop_size, int):
        _size = [crop_size, crop_size]
    elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
        _size = crop_size
    else:
        raise RuntimeError("size error.")
    crop_w, crop_h = _size
    pad_w = max((crop_w - w), 0)
    pad_h = max((crop_h - h), 0)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w // 2, pad_h // 2, (crop_w - w) - pad_w // 2, (crop_h - h) - pad_h // 2)
        image = ImageOps.expand(image, border=border, fill=padding)
        mask = ImageOps.expand(mask, border=border, fill=ignore_label)
    else:
        w_off = (w - crop_w) // 2
        h_off = (h - crop_h) // 2
        border = (w_off, h_off, (w - crop_w) - w_off, (h - crop_h) - h_off)
        image = ImageOps.crop(image, border=border)
        mask = ImageOps.crop(mask, border=border)
    return image, mask


if __name__ == "__main__":
    """
    img = Image.open("/Users/chenhongming/Desktop/segmentation/Semantic_Segmentation_PyTorch/data/Pascal_VOC/VOC2007/JPEGImages/001311.jpg")
    mask = Image.open("/Users/chenhongming/Desktop/segmentation/Semantic_Segmentation_PyTorch/data/Pascal_VOC/VOC2007/SegmentationClass/001311.png")
    r = RandomRotate()
    value = np.unique(mask)
    print('value: ', value)
    w, h = mask.size
    mask_tmp = np.asarray(mask)
    out = np.zeros((h, w, 3)).astype('uint8')
    for i in value:
        out[np.where(mask_tmp == i)] = color_map[i]
    cv.imshow('mask', out)
    cv.waitKey(0)

    t_image, t_mask = r(img, mask)
    value = np.unique(t_mask)
    print('value: ', value)
    out1 = np.zeros((h, w, 3)).astype('uint8')
    for i in value:
        out1[np.where(np.asarray(t_mask) == i)] = color_map[i]
    cv.imshow('t_mask', np.asarray(out1))
    cv.waitKey(0)
    """

    # crop_size = (300, 300)
    # image_list = os.listdir("/Users/chenhongming/Desktop/image/")
    # for i in image_list:
    #     if not i.startswith("."):
    #         image = Image.open("/Users/chenhongming/Desktop/image/" + i)
    #         print("ori image size", image.size)
    #         image, mask = unified_size(image, image, crop_size)
    #         print("image size", image.size)
    #         image.save("/Users/chenhongming/Desktop/image/" + i.split(".")[0] + "_d.jpg")
    #         image.show()
    #         # cv.imshow('image', np.asarray(image))
    #         # cv.waitKey(0)

    x = torch.rand(2, 3, 512, 512)
    # model = torch.load("../ckpts/voc/voc_fcn32s/fcn32s_vgg19_bn_ade20k_Epoch_20_model.pth", map_location=torch.device('cpu'))
    # model_ = model['state_dict']
    # y = model_(x)
    #
    # g = make_dot(y)
    # g.render('model', view=False)