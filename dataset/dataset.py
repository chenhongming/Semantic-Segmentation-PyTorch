import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config.config import cfg
from utils.utils import unified_size, root_path


class JsonDataset(Dataset):
    def __init__(self, root="data/", json_path="", split='train', batch_size=1, crop_size=(640, 512), padding=(0, 0, 0),
                 ignore_label=255, transform=None, augmentations=None):
        self.root = root_path() + root
        self.json_path = json_path
        self.split = split
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
        if (self.split == 'train' or self.split == 'val') and self.batch_size > 1:
            image, mask = unified_size(image, mask, self.crop_size, self.padding, self.ignore_label)
        if self.transform is not None:
            image = self.transform(image)
        # down sample mask(e.g. 8x)
        w, h = mask.size
        mask = mask.resize((w // cfg.MODEL.OUTPUT_STRIDE, h // cfg.MODEL.OUTPUT_STRIDE), Image.NEAREST)
        # convert image & mask to tensor
        image = torch.FloatTensor(image)
        mask = torch.LongTensor(np.array(mask).astype('int32'))
        return image, mask

    def __len__(self):
        return len(self.json_file)
