import os
import json
import glob

from utils.registry import Registry
"""
The class processes some publicly available semantic segmentation datasets 
and generates corresponding json files.

available semantic segmentation datasets: [cityscapes, ade20k, voc, voc_aug, camvid, kitti, mscoco, lip, mapillary]

The format of the json file is: [image_path mask_path,
                                 ...        ...      ]
"""

DATASET_REGISTRY = Registry('dataset')

class GenerateJson:
    def __init__(self, root='../', split='train', check_pairs=[]):
        self.root = root
        self.split = split
        self.check_pairs = check_pairs

    def write_json(self, lst, json_path):
        json_file = os.path.join(json_path, self.split + '.json')
        with open(json_file, 'w') as f:
            json.dump(lst, f, indent=4)

    @DATASET_REGISTRY.register()
    def cityscapes2json(self, json_path="../Cityscapes/"):
        assert self.split in ('train', 'val', 'test')
        if self.split == 'test':
            images_path = glob.glob(self.root + "Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/" +
                                    self.split + "/*/*leftImg8bit.png")
            print("images:", len(images_path))
            self.check_pairs = [item[len(root):] for item in images_path]
        else:
            images_path = glob.glob(self.root + "Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/" +
                                    self.split + "/*/*leftImg8bit.png")
            masks_path = glob.glob(self.root + "Cityscapes/gtFine_trainvaltest/gtFine/" +
                                   self.split + "/*/*gtFine_labelIds.png")
            print("images:", len(images_path), " masks:", len(masks_path))
            assert len(images_path) == len(masks_path)
            for image_path in images_path:
                mask_path = image_path.replace('images', 'gtFine').replace('leftImg8bit', 'gtFine_labelTrainIds')
                self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    def ade20k2json(self, json_path="../ADEChallengeData2016/"):
        assert self.split in ('train', 'val')
        if self.split == 'train':
            sp = 'training'
        else:
            sp = 'validation'
        images_path = glob.glob(self.root + "ADEChallengeData2016/images/" + sp + '/*.jpg')
        masks_path = glob.glob(self.root + "ADEChallengeData2016/annotations/" + sp + '/*.png')
        print("images:", len(images_path), " masks:", len(masks_path))
        assert len(images_path) == len(masks_path)
        for image_path in images_path:
            mask_path = image_path.replace('images', 'annotations').replace('jpg', 'png')
            self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    def voc2json(self, json_path="../Pascal VOC/"):
        assert self.split in ('train', 'val', 'trainval')
        year = ['VOC2007', 'VOC2012']
        for i in year:
            target_file = self.root + "Pascal VOC/" + i + "/ImageSets/Segmentation/" + self.split + ".txt"
            list_sample = [x.rstrip() for x in open(target_file, 'r')]
            for item in list_sample:
                image_path = self.root + "Pascal VOC/" + i + "/JPEGImages/" + item + ".jpg"
                mask_path = self.root + "Pascal VOC/" + i + "/SegmentationClass/" + item + "_mask.png"
                if os.path.isfile(image_path) and os.path.isfile(mask_path):
                    self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
                else:
                    raise Exception("{} or {} is not a file".format(image_path, mask_path))
        print('images:', len(self.check_pairs), 'masks:', len(self.check_pairs))
        self.write_json(self.check_pairs,json_path)

    @DATASET_REGISTRY.register()
    def voc_aug2json(self, json_path="../Pascal VOC_aug/benchmark_RELEASE/dataset"):
        assert self.split in ('train', 'val')
        target_file = self.root + "Pascal VOC_aug/benchmark_RELEASE/dataset/" + self.split + ".txt"
        list_sample = [x.rstrip() for x in open(target_file, 'r')]
        for item in list_sample:
            image_path = self.root + "Pascal VOC_aug/benchmark_RELEASE/dataset/img/" + item + ".jpg"
            mask_path = self.root + "Pascal VOC_aug/benchmark_RELEASE/dataset/seg/" + item + ".png"
            if os.path.isfile(image_path) and os.path.isfile(mask_path):
                self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
            else:
                raise Exception("{} or {} is not a file".format(image_path, mask_path))
        print('images:', len(self.check_pairs), 'masks:', len(self.check_pairs))
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    def camvid2json(self, json_path="../Camvid/data/Camvid/"):
        assert self.split in ('train', 'val', 'test')
        if self.split == 'test':
            images_path = glob.glob(self.root + "Camvid/data/Camvid/" + self.split + "/*.png")
            print("images:", len(images_path))
            self.check_pairs = [item[len(root):] for item in images_path]
        else:
            images_path = glob.glob(self.root + "Camvid/data/Camvid/" + self.split + "/*.png")
            masks_path = glob.glob(self.root + "Camvid/data/Camvid/" + self.split + 'annot/' + "/*.png")
            print("images:", len(images_path), " masks:", len(masks_path))
            assert len(images_path) == len(masks_path)
            for image_path in images_path:
                mask_path = image_path.replace(self.split, self.split + 'annot/')
                if os.path.isfile(mask_path):
                    self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
                else:
                    raise Exception("{} is not a file".format(mask_path))
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    def kitti2json(self, json_path="../KITTI/data_semantics/"):
        assert self.split in ('train', 'test')
        if self.split == 'train':
            sp = 'training'
        else:
            sp = 'testing'
        if self.split == 'test':
            images_path = glob.glob(self.root + "KITTI/data_semantics/" + sp + "/image/*.png")
            print("images:", len(images_path))
            self.check_pairs = [item[len(root):] for item in images_path]
        else:
            images_path = glob.glob(self.root + "KITTI/data_semantics/" + sp + "/image/*.png")
            masks_path = glob.glob(self.root + "KITTI/data_semantics/" + sp  + "/semantic/*.png")
            print("images:", len(images_path), " masks:", len(masks_path))
            assert len(images_path) == len(masks_path)
            for image_path in images_path:
                mask_path = image_path.replace('image', 'semantic')
                if os.path.isfile(mask_path):
                    self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
                else:
                    raise Exception("{} is not a file".format(mask_path))
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    def mscoco2json(self, json_path="../MSCOCO/"):
        assert self.split in ('train', 'val')
        if self.split == 'train':
            sp = 'train2017'
        else:
            sp = 'val2017'
        images_path = glob.glob(self.root + "MSCOCO/" + sp + "/*.jpg")
        masks_path = glob.glob(self.root + "MSCOCO/SegmentationClass/" + sp + "/*.png")
        print("images:", len(images_path), " masks:", len(masks_path))
        assert len(images_path) == len(masks_path)
        for image_path in images_path:
            mask_path = image_path.replace('MSCOCO/', "MSCOCO/SegmentationClass/").replace('jpg', 'png')
            if os.path.isfile(mask_path):
                self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
            else:
                raise Exception("{} is not a file".format(mask_path))
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    def lip2json(self, json_path="../LIP/", extra=True):
        assert self.split in ('train', 'val', 'test')
        _type = ['Multi-Person', 'Single-Person']
        for i in _type:
            if self.split == 'test':
                images_path = glob.glob(self.root + "LIP/" + i + "-LIP/LIP/Testing_images/*.jpg")
                print("images:", len(images_path))
                self.check_pairs = [item[len(root):] for item in images_path]
            else:
                images_path = glob.glob(self.root + "LIP/" + i + "-LIP/LIP/TrainVal_images/" +
                                        self.split +  "_images/*.jpg")
                masks_path = glob.glob(self.root + "LIP/" + i + "-LIP/LIP/TrainVal_parsing_annotations/" +
                                       self.split +  "_segmentations/*.png")
                print("images:", len(images_path), " masks:", len(masks_path))
                assert len(images_path) == len(masks_path)
                for image_path in images_path:
                    mask_path = image_path.replace('TrainVal_images/', "TrainVal_parsing_annotations/").\
                        replace('_images', '_segmentations').replace('jpg', 'png')
                    if os.path.isfile(mask_path):
                        self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
                    else:
                        raise Exception("{} is not a file".format(mask_path))
        if extra and self.split == 'train':
            for i in _type:
                images_path = glob.glob(self.root + "LIP/" + i + "-LIP/ATR/JPEGImages/*.jpg")
                masks_path = glob.glob(self.root + "LIP/" + i + "-LIP/ATR/SegmentationClassAug/*.png")
                print("images:", len(images_path), " masks:", len(masks_path))
                assert len(images_path) == len(masks_path)
                for image_path in images_path:
                    mask_path = image_path.replace('JPEGImages/', "SegmentationClassAug/").replace('jpg', 'png')
                    if os.path.isfile(mask_path):
                        self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
                    else:
                        raise Exception("{} is not a file".format(mask_path))
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    def mapillary2json(self, json_path="../mapillary_vistas_v2_part/"):
        assert self.split in ('train', 'val', 'test')
        if self.split == 'test':
            images_path = glob.glob(self.root + "mapillary_vistas_v2_part/" + self.split + "/images/*.jpg")
            print("images:", len(images_path))
            self.check_pairs = [item[len(root):] for item in images_path]
        else:
            images_path = glob.glob(self.root + "mapillary_vistas_v2_part/" + self.split + "/images/*.jpg")
            masks_path = glob.glob(self.root + "mapillary_vistas_v2_part/" + self.split + "/labels/*.png")
            print("images:", len(images_path), " masks:", len(masks_path))
            assert len(images_path) == len(masks_path)
            for image_path in images_path:
                mask_path = image_path.replace('images/', "labels/").replace('jpg', 'png')
                if os.path.isfile(mask_path):
                    self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
                else:
                    raise Exception("{} is not a file".format(mask_path))
        self.write_json(self.check_pairs, json_path)

    @DATASET_REGISTRY.register()
    # used to generate priv dataset
    def priv2json(self, json_path="../priv/"):
        assert self.split in ('train', 'val', 'test')
        if self.split == 'test':
            images_path = glob.glob(self.root + "priv/images/" + self.split + "/*.jpg")
            print("images:", len(images_path))
            self.check_pairs = [item[len(root):] for item in images_path]
        else:
            images_path = glob.glob(self.root + "priv/images/" + self.split + "/*.jpg")
            masks_path = glob.glob(self.root + "priv/segmentations/" + self.split + "/*.png")
            print("images:", len(images_path), " masks:", len(masks_path))
            assert len(images_path) == len(masks_path)
            for image_path in images_path:
                mask_path = image_path.replace('images', 'segmentations').replace('jpg', 'png')
                if os.path.isfile(mask_path):
                    self.check_pairs.append(image_path[len(root):] + ' ' + mask_path[len(root):])
                else:
                    raise Exception("{} is not a file".format(mask_path))
        self.write_json(self.check_pairs, json_path)


if __name__ == '__main__':
    generate_json = GenerateJson(split='train')

    # registry method is more concise
    DATASET_REGISTRY.get('ade20k2json')(generate_json)

    # -------------------------------------
    # 'class calls' method is more tedious
    # generate_json.cityscapes2json()
    # generate_json.ade20k2json()
    # generate_json.voc2json()
    # generate_json.voc_aug2json()
    # generate_json.camvid2json()
    # generate_json.kitti2json()
    # generate_json.mscoco2json()
    # generate_json.lip2json(extra=True)
    # generate_json.mapillary2json()
    # generate_json.priv2json()
    # -------------------------------------