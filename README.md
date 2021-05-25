# PCNet-C Experiments on COCOA Dataset

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## Contents
1. [Overview](#1-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Experiments](#3-experiments)

## 1. Overview

This repo contains the code for my experiments on **content completion** using the PCNet-C model proposed in [Self-Supervised Scene De-occlusion](https://xiaohangzhan.github.io/projects/deocclusion/).

## 2. Setup Instructions

- Clone the repo:

```shell
https://github.com/praeclarumjj3/PCNet-C-Experiments.git
cd PCNet-C-Experiments
```
- Install [Pytorch](https://pytorch.org/get-started/locally/) and other dependencies:

```shell
pip3 install -r requirements.txt
```

- Install pycocotools:
   
```shell
pip3 install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
```

### Dataset Preparation

- Download the **MS-COCO 2014** images and unzip:
```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

- Download the annotations and untar:
``` 
gdown https://drive.google.com/uc?id=0B8e3LNo7STslZURoTzhhMFpCelE
tar -xf annotations.tar.gz
```

- Unzip the files according to the following structure

```
PCNet-C-Experiments
├── data
│   ├── COCOA
│   │   ├── annotations
│   │   ├── train2014
│   │   ├── val2014
```
## Run Demos

Download released models [here](https://drive.google.com/drive/folders/1O89ItVWucCoL_VxIbLM1XLxr9JFfyj_Y?usp=sharing) and put the folder `released` under `deocclusion`.

## 3. Experiments

### Training


1. Download the pre-trained image inpainting model using partial convolution [here](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/README.md) to `pretrains/partialconv.pth`

2. Convert the model to accept 4 channel inputs.

    ```shell
    python tools/convert_pcnetc_pretrain.py
    ```

3. Train (taking COCOA for example).

    ```
    sh experiments/train.sh # you may have to set --nproc_per_node=#YOUR_GPUS
    ```

## Evaluate

* Execute:

    ```shell
    sh tools/test_cocoa.sh
    ```

## Acknowledgement

This repo borrows heavily from [deocclusion](https://github.com/XiaohangZhan/deocclusion).
