import torch.nn as nn


class _PyramidPoolingModule(nn.Module):
    def __init__(self, dim_in,):
        super().__init__()
