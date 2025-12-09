import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np

class FER2013Dataset(Dataset):
    """
    Loads FER2013 CSV file.
    Each row: emotion, pixel array, usage
    """
    def __init__(self, csv_path, usage="Training"):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["Usage"] == usage].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = np.stack([pixels, pixels, pixels], axis=2)

        img = self.transform(img)
        label = int(row["emotion"])

        return {
            "image": img,
            "label_cls": label,
            "eeg": None,
            "audio": None,
            "label_valence": None,
            "label_arousal": None,
            "subject_id": None,
            "dataset": "FER2013"
        }

