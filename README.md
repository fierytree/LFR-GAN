# Introduction

This is the source code of our TOMM 2023 paper "LFR-GAN: Local Feature Refinement based Generative Adversarial Network for Text-to-Image Generation". Please cite the following paper if you use our code.

Zijun Deng, Xiangteng He and Yuxin Peng*, Zijun Deng, Xiangteng He and Yuxin Peng*, "LFR-GAN: Local Feature Refinement based Generative Adversarial Network for Text-to-Image Generation", ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2023.


# Dependencies

- Python 3.7

- CUDA 1.11.0

- PyTorch 1.7.1

- gcc 7.5.0

Run the following commands to install the same dependencies as our experiments.

```bash
$ conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install git+https://github.com/openai/CLIP.git
$ pip install -r requirements.txt
```


# Data Preparation

Download the image data, pretrained models, and parser tool that we used from the [link](https://pan.baidu.com/s/1Q9Vh2JTOTHnsjmKlyqum2g) (password: 2fsx) and unzip them to corresponding folders.


# Generate Image

1. For bird images: `python code/main.py --cfg=code/cfg/eval_bird.yml --lafite=pretrained_models/birds.pkl`

2. For flower images: `python code/main.py --cfg=code/cfg/eval_flower.yml  --lafite=pretrained_models/flower.pkl`


# Train Model

You can also train the models by yourself.
```bash
$ # pretrained_models/bird_netG_epoch_700.pth
$ python code/main_DMGAN.py --cfg code/cfg/bird_DMGAN.yml --gpu 0
$ # pretrained_models/flower_netG_epoch_325.pth
$ python code/main_DMGAN.py --cfg code/cfg/flower_DMGAN.yml --gpu 0
```

# Evaluate


1. For CUB dataset: `CUDA_VISIBLE_DEVICES=6 python calc_metrics.py --metrics=fid50k_full,is50k --data=datasets/birds_train_clip.zip --test_data=datasets/birds_test_clip.zip --lafite=pretrained_models/birds.pkl --alpha=1.3 --beta=0.01`

2. For Oxford102 dataset: `CUDA_VISIBLE_DEVICES=7 python calc_metrics.py --metrics=fid50k_full,is50k --data=datasets/flower_train_clip2.zip --test_data=datasets/flower_test_clip.zip --lafite=pretrained_models/flower.pkl --alpha=4.7 --beta=0.6`


For any questions, feel free to contact us (dengzijun57@gmail.com).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.
