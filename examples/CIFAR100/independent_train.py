import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms

from dml import DistillationTrainer, Learner, build_links

from dml.models import cifar_models
from dml.utils import AverageMeter, WorkerInitializer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--lr", default=0.1)
parser.add_argument("--wd", default=5e-4)
parser.add_argument("--model", default="resnet32")

args = parser.parse_args()
manualSeed = int(args.seed)
model_name = args.model
lr = float(args.lr)
wd = float(args.wd)

# Fix the seed value
set_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Prepare the CIFAR-100 for training
batch_size = 64
num_workers = 10

# Normalization constants for CIFAR-100
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_dataset = torchvision.datasets.CIFAR100(
    root="data", train=True, download=True, transform=train_transform
)
val_dataset = torchvision.datasets.CIFAR100(
    root="data", train=False, download=True, transform=val_transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)

# Prepare for training
max_epoch = 200

optim_setting = {
    "name": "SGD",
    "args": {
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": wd,
        "nesterov": True,
    },
}
scheduler_setting = {
    "name": "CosineAnnealingLR",
    "args": {"T_max": max_epoch, "eta_min": 0.0},
}

num_classes = 100
learners = []


criterions = [nn.CrossEntropyLoss(reduction="none")]
model = getattr(cifar_models, model_name)(num_classes).to(device)
writer = SummaryWriter(f"runs/pre-train/{model_name}")
save_dir = f"checkpoint/pre-train/{model_name}"
optimizer = getattr(torch.optim, str(optim_setting["name"]))(
    model.parameters(), **optim_setting["args"]
)
scheduler = getattr(torch.optim.lr_scheduler, str(scheduler_setting["name"]))(
    optimizer, **scheduler_setting["args"]
)
links = build_links(criterions)

learner = Learner(
    model=model,
    writer=writer,
    scaler=torch.amp.GradScaler(device.type, enabled=(device.type == "cuda")),
    save_dir=save_dir,
    optimizer=optimizer,
    scheduler=scheduler,
    links=links,
    loss_meter=AverageMeter(),
    score_meter=AverageMeter(),
)
learners.append(learner)

trainer = DistillationTrainer(
    learners=learners,
    max_epoch=max_epoch,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
    device=device,
)
trainer.train()
