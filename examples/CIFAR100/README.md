# CIFAR-100 Experiments

This directory contains experimental results for CIFAR-100 classification using various training strategies.

## 1. Independent Training (Baseline)

Base models trained individually without any knowledge transfer.

| Model | Accuracy |
|-------|----------|
| ResNet32 | **71.19%** |
| WideResNet28-2 | **75.58%** |

## 2. Knowledge Distillation (KD)

Standard Knowledge Distillation (Hinton et al.) with Temperature $T=2.0$.

- **Teacher**: WideResNet28-2 (Pre-trained)
- **Student**: ResNet32

| Experiment | Accuracy |
|------------|----------|
| ResNet32 (Student) | **72.17%** |

## 3. Deep Mutual Learning (DML)

### 3.1 DML with 2 Nodes (T=1.0)

Collaborative learning between two models with Temperature $T=1.0$.

- **Node 0**: ResNet32
- **Node 1**: WideResNet28-2

| Model | Accuracy |
|-------|----------|
| Node 0 (ResNet32) | **72.86%** |
| Node 1 (WideResNet28-2) | **76.49%** |

### 3.2 DML with 4 Nodes (T=1.0)

Collaborative learning between four models with Temperature $T=1.0$.

- **Node 0**: ResNet32
- **Node 1**: WideResNet28-2
- **Node 2**: WideResNet28-2
- **Node 3**: ResNet32

| Model | Accuracy |
|-------|----------|
| Node 0 (ResNet32) | **73.58%** |
| Node 1 (WideResNet28-2) | **77.52%** |
| Node 2 (WideResNet28-2) | **77.29%** |
| Node 3 (ResNet32) | **73.32%** |