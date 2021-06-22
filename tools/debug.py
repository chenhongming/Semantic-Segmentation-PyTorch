import torch
import torchvision
import torchvision.models as models

model = models.mobilenet_v2()
print(torch.__version__)
print(torchvision.__version__)

x = torch.rand([2, 3, 65, 65])
o = model(x)
print(o.size())