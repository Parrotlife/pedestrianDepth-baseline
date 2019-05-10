# Baseline of depth estimation for pedestrians using monodepth
![demo.gif animation](readme_images/demo.gif)

This repo uses the pytorch implementation of the amazing work of [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/) for Unsupervised Monocular Depth Estimation.
Original code and paper could be found via the following links:
1. [Original repo](https://github.com/mrharicot/monodepth)
2. [Original paper](https://arxiv.org/abs/1609.03677)

## Baseline principle

This baseline combines the monodepth estimation algorithm with keypoints 2 dedection methods (mask-rcnn and pifpaf) to estimate the depth of pedestrians.

## Dataset
### KITTI

We run this baseline on a 5000 images sub-dataset of Kitti.

## Dataloader
Dataloader assumes the following structure of the folder with train examples (**'data_dir'** argument contains path to that folder):
The folder contains subfolders with following folders "image_02/data" for left images and  "image_03/data" for right images.
Such structure is default for KITTI dataset

Example data folder structure (path to the "kitti" directory should be passed as **'data_dir'** in this example):
```
data
├── kitti
│   ├── 2011_09_26_drive_0001_sync
│   │   ├── image_02
│   │   │   ├─ data
│   │   │   │   ├── 0000000000.png
│   │   │   │   └── ...
│   │   ├── image_03
│   │   │   ├── data
│   │   │   │   ├── 0000000000.png
│   │   │   │   └── ...
│   ├── ...
├── models
├── output
├── test
│   ├── left
│   │   ├── test_1.jpg
│   │   └── ...
```
    
## Training
Example of training can be find in [Monodepth](Monodepth.ipynb) notebook.

Model class from main_monodepth_pytorch.py should be initialized with following params (as easydict) for training:
 - `data_dir`: path to the dataset folder
 - `val_data_dir`: path to the validation dataset folder
 - `model_path`: path to save the trained model
 - `output_directory`: where save dispairities for tested images
 - `input_height`
 - `input_width`
 - `model`: model for encoder (resnet18_md or resnet50_md or any torchvision version of Resnet (resnet18, resnet34 etc.)
 - `pretrained`: if use a torchvision model it's possible to download weights for pretrained model
 - `mode`: train or test
 - `epochs`: number of epochs,
 - `learning_rate`
 - `batch_size`
 - `adjust_lr`: apply learning rate decay or not
 - `tensor_type`:'torch.cuda.FloatTensor' or 'torch.FloatTensor'
 - `do_augmentation`:do data augmentation or not
 - `augment_parameters`:lowest and highest values for gamma, lightness and color respectively
 - `print_images`
 - `print_weights`
 - `input_channels` Number of channels in input tensor (3 for RGB images)
 - `num_workers` Number of workers to use in dataloader

Optionally after initialization, we can load a pretrained model via `model.load`.

After that calling train() on Model class object starts the training process.

Also, it can be started via calling main_monodepth_pytorch.py through the terminal and feeding parameters as argparse arguments.

## Getting monodepth output

We use a pretrained model of monodepth to get our depth estimation.

For training the following parameters were used:
```
`model`: 'resnet18_md'
`epochs`: 200,
`learning_rate`: 1e-4,
`batch_size`: 8,
`adjust_lr`: True,
`do_augmentation`: True
```
The provided model was trained on the whole Kitti dataset, except subsets
    
## Requirements
This code was tested with PyTorch 0.4.1, CUDA 9.1 and Ubuntu 16.04. Other required modules:

```
pifpaf
torchvision
numpy
matplotlib
easydict
```
and also the running of both maskrcnn and pifpaf

## Tutorial
you can use any of the available pedestrian_depth_baseline notebooks.