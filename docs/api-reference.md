# API Reference

Complete API documentation for the Knowledge Transfer Graph library.

## Core Classes

### KnowledgeTransferGraph

Main class for training multiple models collaboratively.

```python
class KnowledgeTransferGraph:
    def __init__(
        self,
        nodes: list[Node],
        max_epoch: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        trial: Optional[optuna.Trial] = None,
    )
```

**Parameters:**
- `nodes` (list[Node]): List of Node instances representing models in the graph
- `max_epoch` (int): Maximum number of training epochs
- `train_dataloader` (DataLoader): DataLoader for training data
- `test_dataloader` (DataLoader): DataLoader for validation/test data
- `trial` (Optional[optuna.Trial]): Optuna trial object for hyperparameter optimization

**Methods:**

#### `train() -> float`
Train all models in the graph and return the best validation score.

**Returns:**
- `float`: Best validation accuracy (from node 0)

**Example:**
```python
graph = KnowledgeTransferGraph(nodes, max_epoch=200, ...)
best_score = graph.train()
```

#### `train_on_batch(image, label, epoch, num_iter)`
Process a single training batch.

**Parameters:**
- `image` (Tensor): Input images
- `label` (Tensor): Ground truth labels
- `epoch` (int): Current epoch (0-indexed)
- `num_iter` (int): Current iteration number

#### `test_on_batch(image, label)`
Process a single validation batch.

**Parameters:**
- `image` (Tensor): Input images
- `label` (Tensor): Ground truth labels

---

### Node

Represents a single model in the knowledge transfer graph.

```python
@dataclass
class Node:
    model: nn.Module
    writer: SummaryWriter
    scaler: torch.amp.GradScaler
    optimizer: Optimizer
    edges: list[Edge]
    loss_meter: AverageMeter
    score_meter: AverageMeter
    scheduler: Optional[LRScheduler] = None
    best_score: float = 0.0
    eval: nn.Module = accuracy
    save_dir: Optional[str] = None
```

**Fields:**
- `model` (nn.Module): The neural network model
- `writer` (SummaryWriter): TensorBoard writer for logging
- `scaler` (torch.amp.GradScaler): Gradient scaler for mixed precision
- `optimizer` (Optimizer): Optimizer for parameter updates
- `edges` (list[Edge]): List of incoming edges (knowledge transfer paths)
- `loss_meter` (AverageMeter): Tracks average training loss
- `score_meter` (AverageMeter): Tracks average accuracy/score
- `scheduler` (Optional[LRScheduler]): Learning rate scheduler
- `best_score` (float): Best validation score achieved
- `eval` (nn.Module): Evaluation function (default: accuracy)
- `save_dir` (Optional[str]): Directory to save checkpoints

**Note:** The `total_loss` field is automatically created from `edges` in `__post_init__` as a `TotalLoss` instance.

---

### Edge

Represents a knowledge transfer path between models.

```python
class Edge(nn.Module):
    def __init__(self, criterion: nn.Module, gate: nn.Module)
```

**Parameters:**
- `criterion` (nn.Module): Loss function (e.g., CrossEntropyLoss, KLDivLoss)
- `gate` (nn.Module): Gate module that controls transfer weight

**Methods:**

#### `forward(target_output, label, source_output, epoch, is_self_edge) -> Tensor`
Compute the weighted loss for this edge.

**Parameters:**
- `target_output` (Tensor): Output from the target model
- `label` (Tensor): Ground truth labels (for self-edges)
- `source_output` (Tensor): Output from the source model (for transfer edges)
- `epoch` (int): Current epoch (0-indexed)
- `is_self_edge` (bool): Whether this is a self-edge

**Returns:**
- `Tensor`: Weighted loss value

---

### build_edges

Helper function to create a list of Edge instances.

```python
def build_edges(
    criterions: list[nn.Module],
    gates: list[nn.Module]
) -> list[Edge]
```

**Parameters:**
- `criterions` (list[nn.Module]): List of loss functions
- `gates` (list[nn.Module]): List of gate modules

**Returns:**
- `list[Edge]`: List of Edge instances

**Broadcasting:**
- If `criterions` has length 1 and `gates` has length N>1, `criterions` is broadcast to length N
- If `gates` has length 1 and `criterions` has length N>1, `gates` is broadcast to length N
- Otherwise, lengths must match

**Example:**
```python
criterions = [CrossEntropyLoss(), KLDivLoss(), KLDivLoss()]
gates = [ThroughGate(200), ThroughGate(200), CutoffGate(200)]
edges = build_edges(criterions, gates)
```

---

## Gates

All gates inherit from `nn.Module` and implement the same interface.

### ThroughGate

Always transfers knowledge (weight = 1.0).

```python
class ThroughGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(self, loss: Tensor, epoch: int) -> Tensor
```

### CutoffGate

Never transfers knowledge (weight = 0.0).

```python
class CutoffGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(self, loss: Tensor, epoch: int) -> Tensor
```

### PositiveLinearGate

Gradually increases transfer from 0 to 1 over epochs.

```python
class PositiveLinearGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(self, loss: Tensor, epoch: int) -> Tensor
```

**Weight formula:** `weight = epoch / (max_epoch - 1)`

### NegativeLinearGate

Gradually decreases transfer from 1 to 0 over epochs.

```python
class NegativeLinearGate(nn.Module):
    def __init__(self, max_epoch: int)
    def forward(self, loss: Tensor, epoch: int) -> Tensor
```

**Weight formula:** `weight = (max_epoch - 1 - epoch) / (max_epoch - 1)`

---

## Loss Functions

### KLDivLoss

Knowledge distillation loss with temperature scaling.

```python
class KLDivLoss(nn.Module):
    def __init__(self, T: float = 1.0)
    def forward(self, y_pred: Tensor, y_gt: Tensor) -> Tensor
```

**Parameters:**
- `T` (float): Temperature parameter (default: 1.0)

**Formula:**
```
y_pred_soft = softmax(y_pred / T)
y_gt_soft = softmax(y_gt.detach() / T)
loss = T^2 * KL(y_gt_soft, y_pred_soft)
```

The `T^2` factor compensates for gradient scaling from temperature scaling. The KL divergence is computed as:
```
kl = teacher * log(teacher / (student + 1e-10) + 1e-10)
loss = T^2 * mean(sum(kl, dim=1))
```

**Example:**
```python
from ktg.losses import KLDivLoss

kl_loss = KLDivLoss(T=4.0)
loss = kl_loss(student_output, teacher_output)
```

---

## Utility Functions

### AverageMeter

Tracks running average of values.

```python
class AverageMeter:
    def __init__(self)
    def update(self, val: float, n: int = 1)
    def reset(self)
    @property
    def avg(self) -> float
```

**Example:**
```python
from ktg.utils import AverageMeter

meter = AverageMeter()
meter.update(0.95, n=100)
meter.update(0.90, n=50)
print(meter.avg)  # Average of all values
```

### save_checkpoint

Save model checkpoint.

```python
def save_checkpoint(
    model: nn.Module,
    save_dir: str,
    epoch: int,
    is_best: bool = False
)
```

**Parameters:**
- `model` (nn.Module): Model to save
- `save_dir` (str): Directory to save checkpoint
- `epoch` (int): Current epoch number
- `is_best` (bool): If True, save as `best_checkpoint.pkl`

### load_checkpoint

Load model checkpoint.

```python
def load_checkpoint(
    model: nn.Module,
    save_dir: str,
    epoch: int = 1,
    is_best: bool = False
)
```

**Parameters:**
- `model` (nn.Module): Model to load weights into
- `save_dir` (str): Directory containing checkpoint
- `epoch` (int): Epoch number (ignored if `is_best=True`)
- `is_best` (bool): If True, load from `best_checkpoint.pkl`

### set_seed

Set random seed for reproducibility.

```python
def set_seed(seed: int)
```

**Parameters:**
- `seed` (int): Random seed value

### accuracy

Compute top-k accuracy.

```python
def accuracy(
    output: Tensor,
    target: Tensor,
    topk: tuple = (1,)
) -> list[float]
```

**Parameters:**
- `output` (Tensor): Model predictions (logits)
- `target` (Tensor): Ground truth labels
- `topk` (tuple): Tuple of k values (e.g., (1, 5))

**Returns:**
- `list[float]`: List of top-k accuracies

### WorkerInitializer

Initialize DataLoader workers with a fixed seed.

```python
class WorkerInitializer:
    def __init__(self, seed: int)
    def worker_init_fn(self, worker_id: int)
```

**Example:**
```python
from ktg.utils import WorkerInitializer

loader = DataLoader(
    dataset,
    worker_init_fn=WorkerInitializer(42).worker_init_fn
)
```

---

## Models

### CIFAR Models

Pre-defined models for CIFAR-10/100 datasets.

```python
from ktg.models import cifar_models

# ResNet variants
model = cifar_models.resnet18(num_classes=10)
model = cifar_models.resnet20(num_classes=10)
model = cifar_models.resnet32(num_classes=10)
model = cifar_models.resnet34(num_classes=10)
model = cifar_models.resnet44(num_classes=10)
model = cifar_models.resnet50(num_classes=10)
model = cifar_models.resnet56(num_classes=10)
model = cifar_models.resnet110(num_classes=10)
model = cifar_models.resnet1202(num_classes=10)

# Wide ResNet
model = cifar_models.wideresnet28_2(num_classes=10)
```

**Parameters:**
- `num_classes` (int): Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)

---

## Datasets

### CIFAR-10 Dataset

```python
from ktg.dataset.cifar_datasets.cifar10 import get_datasets

train_dataset, val_dataset = get_datasets(use_test_mode=False)
```

**Parameters:**
- `use_test_mode` (bool): If `True`, returns (train+val, test) datasets. If `False`, returns (train, val) datasets (default: `False`)

**Returns:**
- `train_dataset`: Training dataset (or train+val if `use_test_mode=True`)
- `val_dataset`: Validation dataset (or test dataset if `use_test_mode=True`)

**Note:** The datasets automatically download CIFAR-10 data to `data/` directory and apply appropriate transforms (normalization, augmentation).

### CIFAR-100 Dataset

```python
from ktg.dataset.cifar_datasets.cifar100 import get_datasets

train_dataset, val_dataset = get_datasets(use_test_mode=False)
```

**Parameters:**
- `use_test_mode` (bool): If `True`, returns (train+val, test) datasets. If `False`, returns (train, val) datasets (default: `False`)

**Returns:**
- `train_dataset`: Training dataset (or train+val if `use_test_mode=True`)
- `val_dataset`: Validation dataset (or test dataset if `use_test_mode=True`)

**Note:** The datasets automatically download CIFAR-100 data to `data/` directory and apply appropriate transforms (normalization, augmentation).

