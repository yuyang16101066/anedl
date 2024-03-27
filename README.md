## [ANEDL: Adaptive Negative Evidential Deep Learning for Open-Set Semi-supervised Learning (AAAI 2024)](https://arxiv.org/pdf/2303.12091.pdf)


This is an PyTorch implementation of ANEDL. If you have any questions, please contact me via yuyang@cse.cuhk.edu.hk




## Requirements
pip install -r requirement.txt

## Usage

### Dataset Preparation

```
mkdir data

```
Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) and put it under ./data.

### Train

Train the model by 50 labeled data per class of CIFAR-100 dataset, 80 known classes:

```
sh train.sh
```

### Evaluation
Evaluate a model trained on cifar100

```
sh test.sh
```

### Trained models

See Result0/cifar100-80-50.pth.tar

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

### Recent Works
There are also some wonderful works focusing on Open-set SSL, like [SSB](https://github.com/YUE-FAN/SSB) and [IOMatch](https://github.com/nukezil/IOMatch). A blog can be found in [Awesome SSL](https://github.com/RabbitBoss/Awesome-Realistic-Semi-Supervised-Learning)
