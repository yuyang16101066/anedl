## [ANEDL: Adaptive Negative Evidential Deep Learning for Open-Set Semi-supervised Learning (AAAI 2024)](https://arxiv.org/pdf/2303.12091.pdf)


This is an PyTorch implementation of ANEDL.
This implementation is based on [OpenMatch](https://github.com/VisionLearningGroup/OP_Match).



## Requirements
pip install -r requirement.txt

## Usage

### Dataset Preparation
Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)


```
mkdir data

```

The dataset should be under ./data.

### Train

Train the model by 50 labeled data per class of CIFAR-100 dataset, 80 known classes:

```
sh train.sh
```

### Evaluation
Evaluate a model trained on cifar100

```
sh test
```

### Trained models
Coming soon.

- [CIFAR10-50-labeled](https://drive.google.com/file/d/1oNWAR8jVlxQXH0TMql1P-c7_i5-taU2T/view?usp=sharing)
- [CIFAR100-50-labeled-55class](https://drive.google.com/file/d/1T5a_p4XUEOexEnjLWpGd-3pme4OzJ2pP/view?usp=sharing)
- ImageNet-30

### Acknowledgement
This repository depends a lot on [OpenMatch](https://github.com/VisionLearningGroup/OP_Match) implementation.
 Thanks for sharing the great code base!

### Reference
If you consider using this code or its derivatives, please consider citing:

```
@article{yu2023adaptive,
  title={Adaptive Negative Evidential Deep Learning for Open-set Semi-supervised Learning},
  author={Yu, Yang and Deng, Danruo and Liu, Furui and Jin, Yueming and Dou, Qi and Chen, Guangyong and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2303.12091},
  year={2023}
}
```
