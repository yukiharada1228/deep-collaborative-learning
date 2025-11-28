# Architecture

This document explains the architecture and design principles of the Knowledge Transfer Graph framework.

## Overview

Knowledge Transfer Graph (KTG) is a framework for training multiple neural networks collaboratively. The framework models knowledge transfer as a directed graph where:

- **Nodes** represent neural network models
- **Edges** represent knowledge transfer paths between models
- **Gates** control the temporal dynamics of knowledge transfer

## Core Components

### 1. KnowledgeTransferGraph

The main class that orchestrates the training of multiple models. It manages:

- Training loop across all nodes
- Batch processing and forward/backward passes
- Evaluation and metric tracking
- Integration with Optuna for hyperparameter optimization

```python
class KnowledgeTransferGraph:
    def __init__(
        self,
        nodes: list[Node],
        max_epoch: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        trial=None,
    )
```

### 2. Node

A Node represents a single model in the graph. Each node contains:

- **model**: The neural network to train
- **edges**: List of edges connecting to this node (incoming edges)
- **optimizer**: Optimizer for updating model parameters
- **scheduler**: Learning rate scheduler
- **writer**: TensorBoard writer for logging
- **scaler**: Gradient scaler for mixed precision training
- **loss_meter**: Tracks average loss
- **score_meter**: Tracks average accuracy/score

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
    scheduler: LRScheduler = None
    best_score: float = 0.0
    eval: nn.Module = accuracy
    save_dir: Optional[str] = None
```

### 3. Edge

An Edge represents a knowledge transfer path. Each edge consists of:

- **criterion**: Loss function for computing the transfer loss
- **gate**: Gate module that controls the transfer weight over time

```python
class Edge(nn.Module):
    def __init__(self, criterion: nn.Module, gate: nn.Module):
        self.criterion = criterion
        self.gate = gate
    
    def forward(self, target_output, label, source_output, epoch, is_self_edge):
        if is_self_edge:
            loss = self.criterion(target_output, label)
        else:
            loss = self.criterion(target_output, source_output)
        return self.gate(loss, epoch)
```

### 4. TotalLoss

Each node has a `TotalLoss` module that aggregates losses from all incoming edges:

```python
class TotalLoss(nn.Module):
    def __init__(self, edges: list[Edge]):
        self.incoming_edges = nn.ModuleList(edges)
    
    def forward(self, model_id, outputs, labels, epoch):
        # Sum losses from all incoming edges
        losses = []
        for i, edge in enumerate(self.incoming_edges):
            if i == model_id:
                # Self-edge: use ground truth label
                loss = edge(target_output, label, None, epoch, True)
            else:
                # Transfer edge: use source model output
                loss = edge(target_output, None, outputs[i], epoch, False)
        return torch.stack(losses).sum()
```

## Training Flow

### Forward Pass

1. For each batch, all models process the input in parallel:
   ```python
   outputs = []
   for node in self.nodes:
       y = node.model(image)
       outputs.append(y)
   ```

2. For each node, compute the total loss:
   ```python
   loss = node.total_loss(model_id, outputs, labels, epoch)
   ```

3. The total loss aggregates:
   - Self-edge loss: CrossEntropyLoss with ground truth labels
   - Transfer edge losses: KLDivLoss with other models' outputs (weighted by gates)

### Backward Pass

1. Scale the loss with gradient scaler (for mixed precision)
2. Backpropagate through the model
3. Update optimizer
4. Update learning rate scheduler

### Evaluation

After each epoch:
1. Evaluate all models on validation set
2. Log metrics to TensorBoard
3. Save best checkpoints
4. Report to Optuna trial (if applicable)

## Gates

Gates control the temporal dynamics of knowledge transfer. They take a loss value and current epoch, and return a weighted loss:

### ThroughGate

Always transfers knowledge:
```python
def forward(self, loss, epoch):
    return loss  # weight = 1.0
```

### CutoffGate

Never transfers knowledge:
```python
def forward(self, loss, epoch):
    return loss.detach() * 0.0  # weight = 0.0
```

### PositiveLinearGate

Gradually increases transfer from 0 to 1:
```python
def forward(self, loss, epoch):
    loss_weight = epoch / (self.max_epoch - 1)
    return loss * loss_weight
```

### NegativeLinearGate

Gradually decreases transfer from 1 to 0:
```python
def forward(self, loss, epoch):
    loss_weight = (self.max_epoch - 1 - epoch) / (self.max_epoch - 1)
    return loss * loss_weight
```

## Loss Functions

### CrossEntropyLoss

Standard classification loss for self-edges (training with ground truth labels).

### KLDivLoss

Knowledge distillation loss for transfer edges. Uses temperature scaling:

```python
class KLDivLoss(nn.Module):
    def __init__(self, T=1):
        self.T = T  # Temperature parameter
    
    def forward(self, y_pred, y_gt):
        y_pred_soft = softmax(y_pred / T)
        y_gt_soft = softmax(y_gt.detach() / T)
        return (T**2) * KL_divergence(y_pred_soft, y_gt_soft)
```

The `T^2` factor compensates for the gradient scaling introduced by temperature scaling.

## Graph Structure

In a typical setup with N nodes:

- Each node has N edges (one to each node, including itself)
- The self-edge uses CrossEntropyLoss
- Other edges use KLDivLoss for knowledge distillation
- Gates can be configured independently for each edge

This creates a fully connected graph where each model can learn from all others.

## Integration with Optuna

KTG supports hyperparameter optimization with Optuna:

1. Pass an Optuna `trial` object to `KnowledgeTransferGraph`
2. Suggest hyperparameters in the objective function (gates, models, etc.)
3. Report validation scores after each epoch
4. Use pruning to stop unpromising trials early

```python
def objective(trial):
    # Suggest hyperparameters
    gate_name = trial.suggest_categorical("gate", ["ThroughGate", "CutoffGate"])
    model_name = trial.suggest_categorical("model", ["resnet32", "resnet110"])
    
    # Create graph
    graph = KnowledgeTransferGraph(..., trial=trial)
    best_score = graph.train()
    return best_score
```

## Design Principles

1. **Modularity**: Each component (Node, Edge, Gate) is independent and replaceable
2. **Flexibility**: Support for arbitrary graph structures and gate configurations
3. **Efficiency**: Parallel forward passes and efficient loss computation
4. **Extensibility**: Easy to add new gates, loss functions, or models

