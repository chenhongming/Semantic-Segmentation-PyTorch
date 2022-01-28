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
*NOTE:*
It is recommended to modify it to a domestic download mirror source.
```
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```  
or 
```
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```  

## Supported Models

### 1. Backbone
- [x]  vgg
- [x]  resnet 
- [x]  resnext
- [x]  densenet 
- [x]  mobilenet_v1
- [x]  mobilenet_v2
- [x]  mobilenet_v3
- [x]  shufflenetv1
- [x]  shufflenetv2
- [x]  mnasnet 
### 2. Head
- [x] fcn_head
- [x] asppv3_head
- [x] lraspp_head
- [x] denseaspp_head
- [x] ppm_head
- [x] BiSe_head

*NOTE:*  Backone and Head can be freely combined to form a segmentation network, see [models/](https://github.com/chenhongming/Semantic-Segmentation-PyTorch/tree/master/models) for details.
## Get Started  

### 1. Dataset Prepare  

* public dataset  

  Support `cityscapes`, `ade20k`, `voc(07+12)`, `voc_aug`, `camvid`, `kitti`, `mscoco`, `lip`, `mapillary` now.  

  *1.* Downloads the dataset and store it in the `data` folder.

   |  dataset    |  link                 |  link                 | 
   | :-----------| :-------------------: | :-------------------: |  
   | cityscapes  | [Official Website](https://www.cityscapes-dataset.com/downloads/) |                       [baiduyun](https://pan.baidu.com/s/1c_1aZFY1TnwW-YYRB5cC3w)  v1rj |  
   | ade20k      | [Official Website](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) |                                                       [baiduyun](https://pan.baidu.com/s/1hjeCzXutn6l7tNwrt38K9Q) 5rvh |  
   | voc(07+12)  | [Official Website(07)](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) [Official Website(12)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)|                                             [baiduyun](https://pan.baidu.com/s/1eGVnir5Zk3GYo2dcC7aiTg) d4kg |  
   | voc_aug     | [Official Website](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) |                                                                           [baiduyun](https://pan.baidu.com/s/1AOgs728a7sEA6KLaOXdPfA) 2p0l | 
   | camvid      | [Official Website](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels) |                                                            [baiduyun](https://pan.baidu.com/s/1g_AluPvJ36fPGkeznptDuA) wwgk |  
   | kitti       | [Official Website](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) |                                                        [baiduyun](https://pan.baidu.com/s/1uT0A0chXlfp8A5x2kX7XbA) hi2g |  
   | mscoco      | [Official Website(train2017)](http://images.cocodataset.org/zips/train2017.zip)            [Official Website(trainval2017)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)      [Official Website(val2017)](http://images.cocodataset.org/zips/val2017.zip)|                                           [baiduyun](https://pan.baidu.com/s/1VbV34N8h3uvkEvwVmk3B9A) om06 |  
   | lip         | [Official Website](http://hcp.sysu.edu.cn/lip) |                                          [baiduyun](https://pan.baidu.com/s/1aa2ykYrwhCxx8yl_UjJieQ) 41qt | 
   | mapillary   | [Official Website](https://www.mapillary.com/dataset/vistas) |                            [baiduyun](https://pan.baidu.com/s/1EI0IlZNWMjCgkwNjb9U36Q) 0wo1 |  
   
    **If you downloaded a dataset from baiduyun , you can skip `step2.Preprocessing dataset` and `step3.Generate json file` . It is recommended to download from baiduyun.**  
    
  *2.* Preprocessing dataset  
  
  For  `cityscapes`, `voc(07+12)`, `voc_aug`,  `mscoco`, we need to convert the annotation information provided on the official website into a mask.  
  
  Modify the path parameters of the corresponding dataset in the  [data/utils/preprocess.py](https://github.com/chenhongming/Semantic-Segmentation-PyTorch/blob/master/data/utils/preprocess.py) , and then run： 
  
  ```
  python3 preprocess.py
  ```
  *3.* Generate json file
  
  Modify `split` and  `*2json` ( * represents the name of the dataset) parameters in the [data/utils/generate_json.py](https://github.com/chenhongming/Semantic-Segmentation-PyTorch/blob/master/data/utils/generate_json.py), run:  
   
  ```
  python3 generate_json.py
  ```
  
  
  *NOTE：*  The json file is stored in the root directory of the dataset.
* priv dataset  

  Store the prepared images and masks in the folder of the following data structure, Modify `split` and  `priv2json` parameters in the [data/utils/generate_json.py](https://github.com/chenhongming/Semantic-Segmentation-PyTorch/blob/master/data/utils/generate_json.py), run:
  
  ```
  python3 generate_json.py
  ```
  

  ```
  ==========data structure==========
  
  priv
  ├── annotations├── train      # folder
  |              └── val        # folder 
  ├── images ├── train          # folder
  |          ├── val            # folder
  |          └── test           # folder for visualization
  ├── train.json                # file
  └── val.json                  # file
  ```

### 2. Pretrained Backbone Models. 

  1. It can be stored in the `pretrained` folder in advance.

  or

  2. During training, the corresponding pretrained model will be downloaded online and saved in the `~/.cache/torch/hub/checkpoints/`folder if [torchvision](https://pytorch.org/vision/stable/models.html#classification) supports it, otherwise the backone model will be initialized by `kaiming_normal`.  
  
  ### 3. Train
  
   Modify `.yaml` file in  `config/voc/` folder. See [../config/config.py](https://github.com/chenhongming/Semantic-Segmentation-PyTorch/tree/master/config/config.py) for all options.  
   
   *NOTE:*  Check  `MODEL:PHASE:'train'`
   
   * Single GPU or CPU 
   ```
   python3 train.py --cfg ../config/voc/voc_fcn32s.yaml 
   ```
  
   * Multi GPU

  *NOTE:* `nproc_per_node=4` means using 4 GPUs.
  ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --cfg ../config/voc/voc_fcn32s.yaml  
   ```
   
   ### 4. Eval
  
   Modify `.yaml` file in  `config/voc/` folder. See [../config/config.py](https://github.com/chenhongming/Semantic-Segmentation-PyTorch/tree/master/config/config.py) for all options.  
   
   *NOTE:*  Check  `MODEL:PHASE:'val'`
   
  * Single GPU or CPU 
   ```
   python3 eval.py --cfg ../config/voc/voc_fcn32s.yaml
   ```
   
   ### 5. Test
  
   Modify `.yaml` file in  `config/voc/` folder. See [../config/config.py](https://github.com/chenhongming/Semantic-Segmentation-PyTorch/tree/master/config/config.py) for all options.  

   
   *NOTE:*  Check  `MODEL:PHASE:'test'`
   
   * Single GPU or CPU 
   ```
   python3 test.py --cfg ../config/voc/voc_fcn32s.yaml 
   ```
   
   ### 6. Performance

   * Doing and Debuging ...  

   ### 7. Reference 
   
   * [semseg](https://github.com/hszhao/semseg)
   * [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
   * [semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation)
   * [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)
   * [pytorch-segmentation](https://github.com/nyoki-mtl/pytorch-segmentation)
   * [pytorch-semantic-segmentation](https://github.com/zijundeng/pytorch-semantic-segmentation)
   * [detectron2](https://github.com/facebookresearch/detectron2)