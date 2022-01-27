import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import unified_size, root_path


class JsonDataset(Dataset):
    def __init__(self, root="data/", dataset="voc", json_path="", batch_size=1, crop_size=(640, 512), padding=(0, 0, 0),
                 ignore_label=-1, transform=None, augmentations=None):
        self.root = root_path() + root
        self.dataset = dataset
        self.json_path = json_path
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.padding = padding
        self.ignore_label = ignore_label
        self.transform = transform
        self.augmentations = augmentations
        with open(self.root + self.json_path, 'r') as f:
            self.json_file = json.load(f)

    def __getitem__(self, index):
        image = Image.open(self.root + self.json_file[index].split(' ')[0]).convert('RGB')
        mask = Image.open(self.root + self.json_file[index].split(' ')[1])
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)
        if self.batch_size > 1:
            image, mask = unified_size(image, mask, self.crop_size, self.padding, self.ignore_label)
        if self.transform is not None:
            image = self.transform(image)
        if self.dataset in ['voc', 'pascal_voc', 'voc_aug']:
            mask = self._mask_transform(mask)
        # convert image & mask to tensor
        image = torch.FloatTensor(image)
        mask = torch.LongTensor(np.array(mask).astype('int32'))
        return image, mask

    # for pascal voc dataset
    @staticmethod
    def _mask_transform(mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return target

    def __len__(self):
        return len(self.json_file)
