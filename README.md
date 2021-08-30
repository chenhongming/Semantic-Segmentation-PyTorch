# Semantic-Segmentation-PyTorch

## Introduce  
A platform for training and testing semantic segmentation models using PyTorch  
## Installation  
### 1. Clone the repository:  
```
git clone https://github.com/chenhongming/Semantic-Segmentation-PyTorch.git
```  
### 2. Dependencies  
* python>=3.6
* torch>=1.8.0
* torchvision>=0.9.0
```
pip3 install -r requirements.txt
```  
#### NOTE  
It is recommended to modify it to a domestic download mirror source.
```
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```  
or 
```
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```  
## Get Started  

### 1. Dataset Prepare  

* public dataset  

  Support `cityscapes`, `ade20k`, `voc`, `voc_aug`, `camvid`, `kitti`, `mscoco`, `lip`, `mapillary` now. 

  We performed simple preprocessing on these public semantic segmentation datasets and converted them into json format.  
  
* priv dataset  

* dataset format

```
dataset name
├── annotations # folder
├── images      # folder
├── train.json  # file
└── val.json    # file
```
