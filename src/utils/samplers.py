#!/usr/bin/env python3
"""
samplers.py

Balanced sampling for emotion datasets.

Provides:
- BalancedClassSampler: ensures equal sampling for each emotion class.
- WeightedRandomSampler wrapper for PyTorch.

To use in DataLoader:
    sampler = BalancedClassSampler(labels)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
"""

import torch
from torch.utils.data import Sampler, WeightedRandomSampler
import numpy as np

class BalancedClassSampler(Sampler):
    """
    Balances datasets for classification by sampling each emotion class equally.
    Only works when labels are 0..num_classes-1 (ignore_index=-1 ignored).
    """
    def __init__(self, labels, ignore_index=-1):
        self.labels = np.array(labels)
        mask = self.labels != ignore_index
        self.valid_idx = np.where(mask)[0]
        valid_labels = self.labels[mask]

        # Count per class
        unique, counts = np.unique(valid_labels, return_counts=True)
        freq = dict(zip(unique, counts))

        # Weight = inverse frequency
        weights = np.zeros_like(self.labels, dtype=float)
        for idx in self.valid_idx:
            lbl = self.labels[idx]
            weights[idx] = 1.0 / freq[lbl]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        # Weighted sampling over all positions but heavily favors minority classes
        sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)
        return iter(sampler)

    def __len__(self):
        return len(self.labels)

def make_balanced_sampler_from_dataset(dataset):
    """
    Utility to create a sampler if dataset returns dicts with "label_cls".
    """
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(item.get("label_cls", -1))

    return BalancedClassSampler(labels)

