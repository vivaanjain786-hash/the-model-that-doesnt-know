"""Train a 5-class CIFAR-10 classifier using ImageNet-pretrained ResNet-18."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

KNOWN_CLASSES = ["airplane", "automobile", "bird", "cat", "deer"]
CLASS_TO_IDX = {name: i for i, name in enumerate(KNOWN_CLASSES)}
SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes=5, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transforms():
    # ImageNet normalization is required for the pretrained ResNet-18 weights.
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf


def make_loaders(data_dir, batch_size, num_workers, val_fraction):
    train_tf, eval_tf = get_transforms()
    train_aug = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    train_plain = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_tf)

    known_ids = [i for i, label in enumerate(train_aug.targets)
                 if train_aug.classes[label] in KNOWN_CLASSES]
    rng = np.random.default_rng(SEED)
    rng.shuffle(known_ids)
    n_val = int(len(known_ids) * val_fraction)
    val_ids, train_ids = known_ids[:n_val], known_ids[n_val:]

    train_ds = Subset(train_aug, train_ids)
    # Use a separate non-augmented dataset for validation.
    val_ds = Subset(train_plain, val_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def remap_labels(labels, original_classes):
    # CIFAR-10's canonical class order is used to map known classes to 0..4.
    lookup = {original_classes.index(name): i for i, name in enumerate(KNOWN_CLASSES)}
    return torch.tensor([lookup[int(y)] for y in labels], dtype=torch.long)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = remap_labels(labels, ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train without ImageNet weights (usually slower/worse).")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: CUDA is not available; training will be slower.")

    train_loader, val_loader = make_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.val_fraction
    )
    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(row)
        print(json.dumps(row))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "known_classes": KNOWN_CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "seed": args.seed,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "architecture": "resnet18",
                "pretrained": not args.no_pretrained,
            }, output_dir / "best_model.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
