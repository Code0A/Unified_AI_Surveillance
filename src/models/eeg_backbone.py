import torch
import torch.nn as nn

class EEGBackbone(nn.Module):
    """
    Simple 1D CNN for EEG feature extraction.
    Input shape: [batch, channels, time]
    """
    def __init__(self, embed_dim=128, in_channels=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.network(x).squeeze(-1)
        x = self.fc(x)
        return x


