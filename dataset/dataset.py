import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class JsonDataset(Dataset):
    def __init__(self, root="../data/", json_path="", split='train', transform=None, augmentations=None):
        self.root = root
        self.json_path = json_path
        self.split = split
        self.transform = transform
        self.augmentations = augmentations
        with open(self.root + self.json_path, 'r') as f:
            self.json_file = json.load(f)

    def __getitem__(self, index):
        image = Image.open(self.root + self.json_file[index].split(' ')[0]).convert('RGB')
        mask = Image.open(self.root + self.json_file[index].split(' ')[1])
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)
        if self.transform is not None:
            image = self.transform(image)
        # convert image & mask to tensor
        image = torch.FloatTensor(image)
        mask = torch.LongTensor(np.array(mask).astype('int32'))
        return image, mask

    def __len__(self):
        return len(self.json_file)
