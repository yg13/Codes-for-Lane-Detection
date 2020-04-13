## Training ERFNet on 3D Lane Synthetic dataset

### Introduction
The original code is based on a pytorch implementation of 
[ERFNET](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/ERFNet-CULane-PyTorch) trained on CULane Dataset.
This repo includes modification to train the erfnet model on the 
[3D lane synthetic dataset](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset).

The trained model can be used as the segmentation subnetwork in a two-stage 3D lane detection framework, which refers to

'Gen-LaneNet: a generalized and scalable approach for 3D lane detection', Y Guo, etal. Arxiv 2020. [[paper](https://arxiv.org/abs/2003.10656)]

### Requirements
- pytorch 1.4.0

### Data preparation

* Put the training and testing files in 'data_splits/standard' from [3D lane synthetic dataset repo](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset)
 in 'list/sim3d', as 'train.json', and 'val.json'
 * Download the [raw datasets](https://drive.google.com/open?id=1Kisxoj7mYl1YyA_4xBKTE8GGWiNZVain). 

The details of generating the training and testing data
split refers to the [3D lane synthetic dataset](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset). This default
data split is used to repeat the experiments in Gen-LaneNet.

### Training

    python main_erfnet_sim3d.py

* Modify 'dataset_folder' to the location saving the downloaded raw dataset. 
* The trained model will be saved as 'trained_sim3d/_erfnet_model_best.pth.tar'
* Segmentation results are saved in 'predicts/sim3d/output'