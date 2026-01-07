import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision
from models import cifar_models
from models.simclr_model import SimCLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dml.utils import (AverageMeter, WorkerInitializer, accuracy,
                       load_checkpoint, save_checkpoint, set_seed)

parser = argparse.ArgumentParser(description="Linear Evaluation for SimCLR on CIFAR-10")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=256, type=int, help="Batch size")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
parser.add_argument("--model", default="resnet18", type=str, help="Model name")
parser.add_argument("--projection-dim", default=128, type=int, help="Projection dim")
parser.add_argument("--wd", default=0.0, type=float, help="Weight decay")
parser.add_argument(
    "--checkpoint",
    required=True,
    type=str,
    help="Path to pretrained SimCLR checkpoint",
)

args = parser.parse_args()
manualSeed = int(args.seed)
lr = float(args.lr)
wd = float(args.wd)
batch_size = args.batch_size
max_epoch = args.epochs
model_name = args.model
projection_dim = args.projection_dim
checkpoint_path = args.checkpoint

print("=" * 60)
print(f"Linear Evaluation: {model_name}")
print("=" * 60)
print(f"Seed: {manualSeed}")
print(f"Learning rate: {lr}")
print(f"Weight decay: {wd}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {max_epoch}")
print(f"Checkpoint: {checkpoint_path}")
print("=" * 60)
print()

# Fix the seed value
set_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print()

# Prepare the CIFAR-10 for evaluation
num_workers = 10

# Normalization constants for CIFAR-10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

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

train_dataset = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=train_transform
)
val_dataset = torchvision.datasets.CIFAR10(
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

num_classes = 10

print("Loading pretrained model...")
print()

# Load pretrained SimCLR model
encoder_func = lambda: getattr(cifar_models, model_name)(num_classes)
simclr_model = SimCLR(encoder_func, out_dim=projection_dim).to(device)

# Load checkpoint using dml utility
load_checkpoint(simclr_model, checkpoint_path)
simclr_model = simclr_model.to(device)
print(f"Loaded checkpoint from: {checkpoint_path}")

# Extract encoder and freeze it
encoder = simclr_model.encoder
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval()

# Get encoder output dimension
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    encoder_output = encoder(dummy_input)
    encoder_dim = encoder_output.shape[1]

print(f"Encoder output dimension: {encoder_dim}")
print()

# Create linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, encoder, encoder_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)


model = LinearClassifier(encoder, encoder_dim, num_classes).to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} (linear classifier only)")
print()

# Setup optimizer and scheduler
optimizer = torch.optim.SGD(
    model.fc.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=wd,
    nesterov=True,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=0.0
)

# Use cross-entropy loss
criterion = nn.CrossEntropyLoss(reduction="mean")
scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

# Setup logging and checkpointing
save_dir = f"checkpoint/linear_eval/{model_name}"
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(f"runs/linear_eval/{model_name}")
best_score = 0.0

print("=" * 60)
print("Starting linear evaluation...")
print("=" * 60)
print()

# Training loop
for epoch in range(1, max_epoch + 1):
    print(f"Epoch {epoch}/{max_epoch}")
    start_time = time.time()

    # Train phase
    train_loss_meter = AverageMeter()
    train_score_meter = AverageMeter()

    model.train()
    for image, label in train_dataloader:
        image = image.to(device)
        label = label.to(device)

        # Forward pass
        with torch.amp.autocast(device_type=device.type):
            output = model(image)
            loss = criterion(output, label)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()

        # Metrics
        [top1] = accuracy(output, label, topk=(1,))
        train_score_meter.update(top1.item(), label.size(0))
        train_loss_meter.update(loss.item(), label.size(0))

    # Log training metrics
    lr_current = optimizer.param_groups[0]["lr"]
    train_loss = train_loss_meter.avg
    train_score = train_score_meter.avg

    writer.add_scalar("train_lr", lr_current, epoch)
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_score", train_score, epoch)

    print(f"  Train: loss={train_loss:.4f}, acc={train_score:.2f}%")

    scheduler.step()

    # Validation phase
    test_score_meter = AverageMeter()

    model.eval()
    for image, label in val_dataloader:
        image = image.to(device)
        label = label.to(device)

        with torch.amp.autocast(device_type=device.type):
            with torch.no_grad():
                output = model(image)

        [top1] = accuracy(output, label, topk=(1,))
        test_score_meter.update(top1.item(), label.size(0))

    # Log validation metrics and save checkpoint
    test_score = test_score_meter.avg
    writer.add_scalar("test_score", test_score, epoch)

    print(f"  Test:  acc={test_score:.2f}%", end="")

    if test_score >= best_score:
        best_score = test_score
        print(" [BEST]")
    else:
        print()

    save_checkpoint(model, save_dir, epoch, filename="latest_checkpoint.pkl")

    elapsed_time = time.time() - start_time
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print()

# Close writer
writer.close()

print("=" * 60)
print("Linear evaluation completed!")
print("=" * 60)
print(f"Best test accuracy: {best_score:.2f}%")
print("=" * 60)
