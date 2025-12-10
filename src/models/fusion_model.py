import torch
import torch.nn as nn
from src.models.visual_backbone import VisualBackbone
from src.models.eeg_backbone import EEGBackbone
from src.models.personalization import SubjectEmbedding

class FusionModel(nn.Module):
    """
    Combines all modalities:
    Image + EEG + Subject Embedding → Multi-task outputs
    """
    def __init__(self, num_classes=7):
        super().__init__()

        self.visual = VisualBackbone(embed_dim=256)
        self.eeg = EEGBackbone(embed_dim=128)

        self.subject_embed = SubjectEmbedding(embed_dim=64)

        fused_dim = 256 + 128 + 64

        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.cls_head = nn.Linear(256, num_classes)
        self.valence_head = nn.Linear(256, 1)
        self.arousal_head = nn.Linear(256, 1)

    def forward(self, image, eeg, subject_id):
        v = self.visual(image)
        e = self.eeg(eeg)
        s = self.subject_embed(subject_id)

        fused = torch.cat([v, e, s], dim=1)
        h = self.fc(fused)

        return {
            "logits": self.cls_head(h),
            "valence": self.valence_head(h).squeeze(),
            "arousal": self.arousal_head(h).squeeze()
        }

