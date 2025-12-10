#!/usr/bin/env python3
"""
train_visual_only.py

Train a ResNet18 classifier on FER2013 (or any image dataset producing 'image' & 'label_cls').

Example:
  python src/train/train_visual_only.py \
    --fer_csv data/processed/fer2013.csv \
    --epochs 10 \
    --batch_size 64 \
    --out_dir outputs/visual
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim import AdamW
from torchvision import transforms

# Import FER loader
from src.datasets.fer_loader import FER2013Dataset

def get_dataloaders(fer_csv, batch_size):
    train_ds = FER2013Dataset(fer_csv, usage="Training")
    val_ds = FER2013Dataset(fer_csv, usage="PublicTest")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def build_model(num_classes=7, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def validate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label_cls"].to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            mask = labels >= 0
            if mask.sum() == 0:
                continue
            total += mask.sum().item()
            correct += (preds[mask] == labels[mask]).sum().item()
    return 0.0 if total == 0 else correct / total

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(args.fer_csv, args.batch_size)

    model = build_model(num_classes=args.num_classes, pretrained=not args.no_pretrain).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_val = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 1):
            imgs = batch["image"].to(device)
            labels = batch["label_cls"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % args.log_interval == 0:
                print(f"[Epoch {epoch}] Step {i}/{len(train_loader)} loss={running_loss / i:.4f}")

        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch} finished. Val Acc: {val_acc:.4f}")

        ckpt_path = Path(args.out_dir) / f"visual_epoch{epoch}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, ckpt_path)

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, Path(args.out_dir) / "visual_best.pt")
            print("Saved new best model.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fer_csv", type=str, required=True, help="Path to FER2013 CSV (processed).")
    p.add_argument("--out_dir", type=str, default="outputs/visual", help="Where to store checkpoints.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--no_pretrain", action="store_true", help="Disable ImageNet pretrained weights.")
    args = p.parse_args()
    train(args)

