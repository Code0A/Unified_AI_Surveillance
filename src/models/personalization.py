import torch
import torch.nn as nn

class SubjectEmbedding(nn.Module):
    """
    Learns a trainable embedding vector for each subject.
    """
    def __init__(self, max_subjects=5000, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(max_subjects, embed_dim)

    def forward(self, subject_ids):
        return self.embed(subject_ids)

