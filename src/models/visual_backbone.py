import torch
import torch.nn as nn
import torchvision.models as models

class VisualBackbone(nn.Module):
    """
    ResNet18-based backbone for face image feature extraction.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_extractor(x)     # [B,512,1,1]
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x

