# Vision Transformer (ViT) Replication on CIFAR-10

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-Library-blue)](https://github.com/huggingface/pytorch-image-models)

这是一个基于 PyTorch 和 `timm` 库实现的 Vision Transformer (ViT) 图像分类项目。本项目在 CIFAR-10 数据集上调试了预训练的 `vit_tiny_patch16_224` 模型。

## 项目关键

- **模型架构**: 使用了经典的 Vision Transformer (ViT-Tiny)。
- **迁移学习**: 利用 ImageNet 预训练权重进行微调，显著提升收敛速度。
- **模块化设计**: 代码结构清晰，区分了数据处理、模型定义与训练流程。
- **实验可视化**: 包含完整的训练 Loss 和测试准确率曲线。

## 目录结构

```text
.
├── data/               # 数据加载与预处理脚本
├── models/             # 模型定义与初始化
├── images/             # 训练结果可视化图表
├── train.py            # 训练主入口
├── requirements.txt    # 环境依赖
└── README.md           # 实验报告
```

## 快速开始

### 1.环境准备

确保已安装 Python 3.8+ 和 PyTorch。
然后安装依赖：

```
pip install -r requirements.txt
```

### 2.开始训练

项目会自动下载 CIFAR-10 数据集并开始训练：

```
python train.py
```

## 实验结果

### 训练配置

```
GPU: NVIDIA Tesla T4\*2 (Kaggle)
Epochs: 3
Batch Size: 64
Learning Rate: 1e-4 (Adam)
Input Size: 224x224 (Resized from 32x32)
```

### 训练曲线

![alt text](./images/training_results.png)

### 性能指标

```
Model : ViT-Tiny (Pretrained)
Dataset : CIFAR-10
Image_Size : 224×224
Accuracy: 95.19%
```

## 实验过程笔记

### 1.输入尺寸的问题

ViT 的 Patch 机制（16x16）要求输入图像不能太小。直接在 CIFAR-10 的 32x32 原图上跑会导致特征丢失，因此必须通过 transforms.Resize(224) 放大图像。

### 2.重复实例化的问题

在实验过程中，注意不要重复实例化模型对象。必须先确定模型结构，再将 model.parameters() 传入优化器，否则模型权重将无法更新。

### 3.收敛速度的问题

得益于 timm 提供的预训练权重，模型在第一个 Epoch 就能达到极高的准确率，展示了迁移学习在小数据集上的强大优势。
