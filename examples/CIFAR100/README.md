# CIFAR-100 Experiments

This directory contains experimental results for CIFAR-100 classification using various training strategies.

## 1. Independent Training (Baseline)

Base models trained individually without any knowledge transfer.

| Model | Accuracy |
|-------|----------|
| ResNet32 | **71.24%** |
| WideResNet28-2 | **75.74%** |

## 2. Knowledge Distillation (KD)

Standard Knowledge Distillation (Hinton et al.) with Temperature $T=2.0$.

- **Teacher**: WideResNet28-2 (Pre-trained)
- **Student**: ResNet32

| Experiment | Accuracy |
|------------|----------|
| ResNet32 (Student) | **71.95%** |

## 3. Deep Mutual Learning (DML)

Collaborative learning between two models with Temperature $T=1.0$.

- **Node 0**: ResNet32
- **Node 1**: WideResNet28-2

| Model | Accuracy |
|-------|----------|
| Node 0 (ResNet32) | **73.05%** |
| Node 1 (WideResNet28-2) | **76.67%** |