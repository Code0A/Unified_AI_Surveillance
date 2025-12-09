data/processed/affectnet/images/*.jpg
data/processed/affectnet/labels.csv
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class AffectNetDataset(Dataset):
    """
    Loads preprocessed AffectNet:
    - Cropped face images
    - CSV labels mapped to unified classes
    """
    def __init__(self, image_folder, labels_csv):
        self.image_folder = image_folder
        self.labels = pd.read_csv(labels_csv)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.image_folder, row["image"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = int(row["emotion"])  # mapped to 0–6 FER classes

        return {
            "image": img,
            "label_cls": label,
            "eeg": None,
            "audio": None,
            "label_valence": None,
            "label_arousal": None,
            "subject_id": row.get("subject_id", None),
            "dataset": "AffectNet"
        }

