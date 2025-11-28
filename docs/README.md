# Knowledge Transfer Graph Documentation

Welcome to the Knowledge Transfer Graph (KTG) documentation. This library implements the "Knowledge Transfer Graph for Deep Collaborative Learning" framework, as described in the [ACCV 2020 paper](https://openaccess.thecvf.com/content/ACCV2020/html/Minami_Knowledge_Transfer_Graph_for_Deep_Collaborative_Learning_ACCV_2020_paper.html).

## Overview

Knowledge Transfer Graph is a framework for training multiple neural networks collaboratively, where each network can learn from others through knowledge transfer. The framework uses a graph structure where nodes represent models and edges control the knowledge transfer between them.

### Key Features

- **Graph-based Architecture**: Flexible graph structure for modeling knowledge transfer between multiple models
- **Temporal Control**: Gates that control knowledge transfer over training epochs
- **Multiple Loss Functions**: Support for various loss functions including KL divergence for knowledge distillation
- **Optuna Integration**: Built-in support for hyperparameter optimization
- **TensorBoard Logging**: Automatic logging of training metrics

## Documentation Structure

- **[Getting Started](getting-started.md)**: Installation and quick start guide
- **[Architecture](architecture.md)**: Detailed explanation of the framework architecture
- **[API Reference](api-reference.md)**: Complete API documentation
- **[Examples](examples.md)**: Usage examples and tutorials

## Quick Example

```python
from ktg import KnowledgeTransferGraph, Node, build_edges, gates
from ktg.losses import KLDivLoss
import torch.nn as nn

# Create nodes (models)
nodes = []
for i in range(3):
    model = YourModel().cuda()
    edges = build_edges(
        criterions=[nn.CrossEntropyLoss() if i == j else KLDivLoss() 
                   for j in range(3)],
        gates=[gates.ThroughGate(max_epoch=200) for _ in range(3)]
    )
    node = Node(
        model=model,
        writer=SummaryWriter(f"runs/node_{i}"),
        scaler=torch.amp.GradScaler("cuda"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(...),
        edges=edges,
        loss_meter=AverageMeter(),
        score_meter=AverageMeter(),
    )
    nodes.append(node)

# Create and train the graph
graph = KnowledgeTransferGraph(
    nodes=nodes,
    max_epoch=200,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
)
best_score = graph.train()
```

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{minami2020knowledge,
  title={Knowledge Transfer Graph for Deep Collaborative Learning},
  author={Minami, Soma and Hirakawa, Tsubasa and Yamashita, Takayoshi and Fujiyoshi, Hironobu},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2020}
}
```

## License

See the [LICENSE](../LICENSE) file for details.

