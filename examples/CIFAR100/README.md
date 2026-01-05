# CIFAR-100 Experiments

This directory contains experimental results for Deep Mutual Learning (DML) on the CIFAR-100 dataset.

## Experimental Setup

### Dataset
- **Dataset**: CIFAR-100
- **Classes**: 100
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Image size**: 32×32

### Data Augmentation
- Random Crop (32×32, padding=4)
- Random Horizontal Flip
- Normalization (mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

### Training Configuration
- **Batch size**: 64
- **Optimizer**: SGD (momentum=0.9, weight_decay=5e-4, nesterov=True)
- **Learning rate**: 0.1
- **Scheduler**: CosineAnnealingLR (T_max=200, eta_min=0.0)
- **Device**: CUDA
- **Seed**: 42

## Experimental Results

### Pre-training (Single Model Baseline)

Baseline experiment with independent training.

**Configuration**:
- Script: [independent_train.py](independent_train.py)
- Epochs: 200
- Model: ResNet-32

**Results**:

| Model | Train Accuracy (%) | Test Accuracy (%) | Train Loss |
|-------|-------------------|-------------------|------------|
| ResNet-32 | 92.83 | 71.19 | 0.265 |

### DML Training with 2 Nodes

Deep Mutual Learning experiment with 2 models.

**Configuration**:
- Script: [dml_train.py](dml_train.py) `--num-nodes 2`
- Epochs: 200
- Models: 2×ResNet-32

**Results**:

| Model ID | Train Accuracy (%) | Test Accuracy (%) | Train Loss | Improvement vs Baseline |
|----------|-------------------|-------------------|------------|------------------------|
| 0_resnet32 | 88.52 | 72.04 | 0.573 | +0.85% |
| 1_resnet32 | 88.92 | 72.19 | 0.563 | +1.00% |
| **Average** | **88.72** | **72.12** | **0.568** | **+0.93%** |

**Observations**:
- Both models achieved test accuracy higher than the baseline
- Training accuracy is lower but test accuracy improved (reduced overfitting)
- Mutual learning between models worked effectively

### DML Training with 3 Nodes (Early Stage)

Deep Mutual Learning experiment with 3 models (in progress).

**Configuration**:
- Script: [dml_train.py](dml_train.py) `--num-nodes 3`
- Epochs: 24/200 (interrupted)
- Models: 3×ResNet-32

**Results**:

| Model ID | Train Accuracy (%) | Test Accuracy (%) | Train Loss |
|----------|-------------------|-------------------|------------|
| 0_resnet32 | 49.73 | 41.24 | 2.130 |
| 1_resnet32 | 50.03 | 46.10 | 2.121 |
| 2_resnet32 | 48.77 | 41.89 | 2.172 |
| **Average** | **49.51** | **43.08** | **2.141** |

**Observations**:
- Early training stage (epoch 24/200)
- Variance observed among models
- Further training required

### DML Training with 4 Nodes (Early Stage)

Deep Mutual Learning experiment with 4 models (in progress).

**Configuration**:
- Script: [dml_train.py](dml_train.py) `--num-nodes 4`
- Epochs: 18/200 (interrupted)
- Models: 4×ResNet-32

**Results**:

| Model ID | Train Accuracy (%) | Test Accuracy (%) | Train Loss |
|----------|-------------------|-------------------|------------|
| 0_resnet32 | 48.68 | 45.31 | 2.156 |
| 1_resnet32 | 48.22 | 44.96 | 2.167 |
| 2_resnet32 | 47.70 | 45.58 | 2.195 |
| 3_resnet32 | 48.35 | 44.88 | 2.172 |
| **Average** | **48.24** | **45.18** | **2.173** |

**Observations**:
- Early training stage (epoch 18/200)
- Relatively stable learning across 4 models
- Performance improvement expected with complete training

## Summary

### Key Findings

1. **Effectiveness of DML (2 nodes)**
   - Approximately 1% improvement in test accuracy compared to baseline
   - Training accuracy decreased but test accuracy improved (better generalization)
   - Confirmed overfitting suppression effect

2. **Impact of Number of Nodes**
   - 2-node DML completed full training and demonstrated effectiveness
   - 3-node and 4-node experiments interrupted during early stages
   - Additional experiments needed to evaluate effectiveness of mutual learning with more nodes

3. **Future Experiments**
   - Complete training for 3-node and 4-node configurations (200 epochs)
   - Try combinations of different architectures
   - Explore other hyperparameters

## Directory Structure

```
examples/CIFAR100/
├── README.md                 # This file
├── dml_train.py             # DML training script
├── independent_train.py     # Independent training script
├── data/                    # CIFAR-100 dataset
├── checkpoint/              # Model checkpoints
│   ├── pre-train/
│   ├── dml_2/
│   ├── dml_3/
│   └── dml_4/
└── runs/                    # TensorBoard logs
    ├── pre-train/
    ├── dml_2/
    ├── dml_3/
    └── dml_4/
```

## How to Run

### Independent Training (Baseline)

```bash
cd examples/CIFAR100
uv run independent_train.py --model resnet32 --lr 0.1 --wd 5e-4 --seed 42
```

### DML Training

```bash
cd examples/CIFAR100

# 2-node DML
uv run dml_train.py --models resnet32 --num-nodes 2 --lr 0.1 --wd 5e-4 --seed 42

# 3-node DML
uv run dml_train.py --models resnet32 --num-nodes 3 --lr 0.1 --wd 5e-4 --seed 42

# 4-node DML
uv run dml_train.py --models resnet32 --num-nodes 4 --lr 0.1 --wd 5e-4 --seed 42
```

### TensorBoard Visualization

```bash
tensorboard --logdir=examples/CIFAR100/runs
```

## References

- Zhang, Y., Xiang, T., Hospedales, T. M., & Lu, H. (2018). Deep Mutual Learning. In CVPR.
