#!/usr/bin/env python3
"""
metrics.py

Utility functions for:
- Classification accuracy / F1
- Valence-arousal regression metrics (RMSE, MAE)
- Exponential moving average smoothing
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

def accuracy(preds, targets, ignore_index=-1):
    """
    preds: [B], predicted class indices
    targets: [B], true class indices
    """
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

def f1_macro(preds, targets, num_classes=7, ignore_index=-1):
    """
    Computes macro F1 across classes.
    """
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    preds = preds[mask].cpu().numpy()
    targets = targets[mask].cpu().numpy()
    try:
        return f1_score(targets, preds, average="macro", labels=list(range(num_classes)))
    except:
        return 0.0

def rmse(pred, target):
    """
    Root Mean Square Error for valence/arousal
    pred & target: tensors of shape [B]
    """
    return torch.sqrt(F.mse_loss(pred, target)).item()

def mae(pred, target):
    """
    Mean absolute error
    """
    return F.l1_loss(pred, target).item()

def ema_smooth(values, alpha=0.2):
    """
    Exponential Moving Average smoothing for realtime predictions.
    values: list or deque of ints (class indices)
    Returns smoothed class score.
    """
    if len(values) == 0:
        return None
    v = np.array(values, dtype=float)
    ema = v[0]
    for x in v[1:]:
        ema = alpha * x + (1 - alpha) * ema
    return int(round(ema))

