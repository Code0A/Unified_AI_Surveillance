data/processed/deap/eeg/
data/processed/deap/labels.csv
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DEAPDataset(Dataset):
    """
    Loads preprocessed DEAP EEG dataset.
    Expected folder:
    data/processed/deap/eeg/<sample_id>.npy
    labels file with: sample_id, valence, arousal, subject_id
    """
    def __init__(self, eeg_folder, labels_csv):
        self.eeg_folder = eeg_folder
        self.labels = pd.read_csv(labels_csv)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        sample_id = row["sample_id"]
        eeg_path = os.path.join(self.eeg_folder, f"{sample_id}.npy")

        eeg = np.load(eeg_path)  # shape: channels × time
        eeg = torch.tensor(eeg, dtype=torch.float32)

        return {
            "image": None,
            "eeg": eeg,
            "audio": None,
            "label_cls": None,
            "label_valence": float(row["valence"]),
            "label_arousal": float(row["arousal"]),
            "subject_id": int(row["subject_id"]),
            "dataset": "DEAP"
        }

