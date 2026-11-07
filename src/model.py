import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size=hidden_size
        self.conv = nn.Conv1d(
            in_channels=20,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, features):
        x = features.transpose(1, 2)
        x = self.relu(self.conv(x))

        return x.transpose(1, 2)


class ProteinClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size = 16):
        super().__init__()

        self.block = ConvBlock(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, features, lengths):
        x = self.block(features)

        positions = torch.arange(x.size(1), device=x.device)
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        x = x * mask.unsqueeze(-1)
        pooled = x.sum(dim=1) / lengths.unsqueeze(1)

        return self.classifier(pooled)
