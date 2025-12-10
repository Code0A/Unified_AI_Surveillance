#!/usr/bin/env python3
"""
train_multimodal.py

Train the multimodal FusionModel on a combination of datasets using UnifiedDataset.

Example:
  python src/train/train_multimodal.py \
    --fer_csv data/processed/fer2013.csv \
    --deap_labels data/processed/deap/labels.csv \
    --deap_eeg data/processed/deap/eeg \
    --epochs 20 \
    --batch_size 32 \
    --out_dir outputs/multimodal
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# dataset loaders
from src.datasets.fer_loader import FER2013Dataset
from src.datasets.deap_loader import DEAPDataset
from src.datasets.unified_dataset import UnifiedDataset, multimodal_collate_fn

# model
from src.models.fusion_model import FusionModel

def make_unified_dataset(fer_csv, deap_eeg_folder, deap_labels_csv):
    fer_train = FER2013Dataset(fer_csv, usage="Training")
    # load DEAP dataset (expects preprocessed .npy + labels csv)
    deap = DEAPDataset(eeg_folder=deap_eeg_folder, labels_csv=deap_labels_csv)
    unified = UnifiedDataset([fer_train, deap])
    return unified

def train_one_epoch(model, loader, optimizer, device, loss_weights):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs = batch["image"].to(device)
        eeg = batch["eeg"].to(device)
        subs = batch["subject_id"].to(device).long()
        cls_targets = batch["label_cls"].to(device)
        val_targets = batch["label_valence"].to(device).float()
        aro_targets = batch["label_arousal"].to(device).float()

        # forward
        out = model(image=imgs, eeg=eeg, subject_id=subs)
        loss_cls = nn.CrossEntropyLoss(ignore_index=-1)(out["logits"], cls_targets)
        loss_val = nn.MSELoss()(out["valence"], val_targets)
        loss_aro = nn.MSELoss()(out["arousal"], aro_targets)

        loss = loss_weights["cls"] * loss_cls + loss_weights["val"] * loss_val + loss_weights["aro"] * loss_aro

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            eeg = batch["eeg"].to(device)
            subs = batch["subject_id"].to(device).long()
            cls_targets = batch["label_cls"].to(device)

            out = model(image=imgs, eeg=eeg, subject_id=subs)
            preds = out["logits"].argmax(dim=1)
            mask = cls_targets >= 0
            if mask.sum() == 0:
                continue
            total += mask.sum().item()
            correct += (preds[mask] == cls_targets[mask]).sum().item()
    return 0.0 if total == 0 else correct / total

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unified = make_unified_dataset(args.fer_csv, args.deap_eeg, args.deap_labels)
    loader = DataLoader(unified, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=multimodal_collate_fn, pin_memory=True)

    # Simple val set: using the FER PublicTest dataset for image validation
    val_ds = FER2013Dataset(args.fer_csv, usage="PublicTest")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = FusionModel(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    loss_weights = {"cls": args.w_cls, "val": args.w_val, "aro": args.w_aro}
    os.makedirs(args.out_dir, exist_ok=True)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device, loss_weights)
        val_acc = validate(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        # save
        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optimizer": optimizer.state_dict(), "val_acc": val_acc}
        torch.save(ckpt, Path(args.out_dir) / f"multimodal_epoch{epoch}.pt")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ckpt, Path(args.out_dir) / "multimodal_best.pt")
            print("Saved new best multimodal model (val acc improved).")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fer_csv", type=str, required=True, help="Path to FER2013 CSV (processed)")
    p.add_argument("--deap_eeg", type=str, required=True, help="Folder containing DEAP eeg .npy files")
    p.add_argument("--deap_labels", type=str, required=True, help="CSV with DEAP labels (sample_id,valence,arousal,subject_id)")
    p.add_argument("--out_dir", type=str, default="outputs/multimodal")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--w_cls", type=float, default=1.0)
    p.add_argument("--w_val", type=float, default=0.5)
    p.add_argument("--w_aro", type=float, default=0.5)
    args = p.parse_args()
    train(args)

