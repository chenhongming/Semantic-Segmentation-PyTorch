import os
import glob
import tqdm
import scipy.io
import imgviz
import cv2 as cv
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

################################
# valid mask file for training #
################################

"""
cityscapes: 34 categories are mapped into 19 categories for training
"""
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                 23, 24, 25, 26, 27, 28, 31, 32, 33]
_key = np.array([-1, -1, -1, -1, -1, -1,
                -1, -1, 0, 1, -1, -1,
                2, 3, 4, -1, -1, -1,
                5, -1, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15,
                -1, -1, 16, 17, 18])
_mapping = np.array(range(-1, len(_key) - 1)).astype('int32')


def _class_to_index(mask):
    # assert the value
    values = np.unique(mask)
    for value in values:
        assert (value in _mapping)
    index = np.digitize(mask.ravel(), _mapping, right=True)
    new_mask = _key[index].reshape(mask.shape)
    return np.where(new_mask == -1, 255, new_mask)


def cityscapes():
    split = 'val'
    masks_path = glob.glob("../Cityscapes/gtFine_trainvaltest/gtFine/" +
                           split + "/*/*gtFine_labelIds.png")
    for mask_path in masks_path:
        mask = cv.imread(mask_path)
        cv.imwrite(mask_path.replace('gtFine_labelIds', 'gtFine_labelTrainIds'), _class_to_index(mask))
    print('Done!')


"""
pascal voc: rgb2mask for training
pascal voc_aug: mat2mask for training
"""


def pascal_palette():
    palette = {(0,     0,   0): 0,
               (128,   0,   0): 1,
               (0,   128,   0): 2,
               (128, 128,   0): 3,
               (0,     0, 128): 4,
               (128,   0, 128): 5,
               (0,   128, 128): 6,
               (128, 128, 128): 7,
               (64,    0,   0): 8,
               (192,   0,   0): 9,
               (64,  128,   0): 10,
               (192, 128,   0): 11,
               (64,    0, 128): 12,
               (192,   0, 128): 13,
               (64,  128, 128): 14,
               (192, 128, 128): 15,
               (0,    64,   0): 16,
               (128,  64,   0): 17,
               (0,   192,   0): 18,
               (128, 192,   0): 19,
               (0,    64, 128): 20}

    return palette


def convert_from_mask_segmentation(seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = pascal_palette()

    for c, i in palette.items():
        color_seg[seg == i] = c

    color_seg = color_seg[..., ::-1]

    return color_seg


def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = pascal_palette()
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


def voc():
    # VOC2012 VOC2007
    rgb_path = glob.glob("../Pascal VOC/VOC2012/SegmentationClass/*.png")
    for item in rgb_path:
        bgr = cv.imread(item)
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        if len(rgb.shape) > 2:
            mask = convert_from_color_segmentation(rgb)
            cv.imwrite(item.replace('.png', '_mask.png'), mask)
        else:
            raise Exception("{} is not composed of three dimensions".format(item))
    print('Done!')


def mat2png(mat_file, key='GTcls'):
    # 'GTcls' key is for class segmentation
    mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
    return mat[key].Segmentation


def convert_mat2png(mat_files, output_path):
    if not mat_files:
        help('Input directory does not contain any Matlab files!\n')

    for mat in mat_files:
        numpy_img = mat2png(mat)
        pil_img = Image.fromarray(numpy_img)
        pil_img.save(os.path.join(output_path, mat.split('/')[-1].replace('.mat', '.png')))


def voc_aug():
    input_path = "../Pascal VOC_aug/benchmark_RELEASE/dataset/cls/"  # cls: semantic label
    output_path = '../Pascal VOC_aug/benchmark_RELEASE/dataset/seg/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.isdir(input_path) and os.path.isdir(output_path):
        mat_files = glob.glob(os.path.join(input_path, '*.mat'))
        convert_mat2png(mat_files, output_path)
        print('Done!')
    else:
        help('Input or output path does not exist!\n')


"""
mscoco 2017: json2mask for training
"""


def save_colored_mask(mask, save_path, mode):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode=mode)
    if mode == "P":
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def mscoco():
    split = 'val'
    mode = "L"  # "P" or "L" https://blog.csdn.net/oYeZhou/article/details/111934432
    annotation_file = os.path.join("../MSCOCO/annotations/", 'instances_{}2017.json'.format(split))
    seg_folder = "../MSCOCO/SegmentationClass/" + '{}2017/'.format(split)
    if not os.path.exists(seg_folder):
        os.makedirs(seg_folder)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            seg_output_path = os.path.join(seg_folder, img['file_name'].replace('.jpg', '.png'))
            save_colored_mask(mask, seg_output_path, mode)
    print('Done!')


def main():
    # cityscapes()
    # voc()
    # voc_aug()
    mscoco()


if __name__ == "__main__":
    main()