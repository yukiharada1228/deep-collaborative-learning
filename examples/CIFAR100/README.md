# CIFAR-100 Experiments

Experiments on the CIFAR-100 dataset.

## Usage

### Independent Training
```bash
uv run independent_train.py --model resnet32
```

### DML Training (2 Nodes)
```bash
uv run dml_train.py --models resnet32 --num-nodes 2
```

## Training Configuration

- **Epochs**: 200
- **Batch size**: 64
- **Optimizer**: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **Scheduler**: CosineAnnealingLR

## Results

### Independent Training

| Model | Test Acc |
|-------|----------|
| ResNet-32 | 71.19% |

### DML Training

#### 2 Nodes

| Node | Test Acc |
|------|----------|
| 0 | 72.04% |
| 1 | 72.19% |
| **Average** | **72.12%** |

#### 3 Nodes

| Node | Test Acc |
|------|----------|
| 0 | 72.52% |
| 1 | 73.31% |
| 2 | 72.10% |
| **Average** | **72.64%** |

#### 4 Nodes

| Node | Test Acc |
|------|----------|
| 0 | 73.11% |
| 1 | 72.99% |
| 2 | 73.63% |
| 3 | 73.10% |
| **Average** | **73.21%** |

#### 5 Nodes (In Progress - 15/200 epochs)

| Node | Test Acc |
|------|----------|
| 0 | 39.16% |
| 1 | 41.02% |
| 2 | 43.38% |
| 3 | 43.63% |
| 4 | 42.31% |
| **Average** | **41.90%** |

#### 6 Nodes (In Progress - 12/200 epochs)

| Node | Test Acc |
|------|----------|
| 0 | 45.17% |
| 1 | 41.62% |
| 2 | 44.89% |
| 3 | 44.13% |
| 4 | 42.92% |
| 5 | 41.82% |
| **Average** | **43.42%** |

## TensorBoard Logs

```bash
tensorboard --logdir examples/CIFAR100/runs
```

Log locations:
- Independent training: `runs/pre-train/resnet32/`
- DML training: `runs/dml_2/{node_id}_resnet32/`
