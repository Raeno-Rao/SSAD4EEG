#models/encoder.py
import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    def __init__(self, in_channels=22, hidden_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=25, padding=12),
            nn.InstanceNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(4),
        )

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
        )

    def forward(self, x):
        """
        x: (B,C,T)
        """

        x = self.conv(x)

        x = x.transpose(1, 2)

        out, _ = self.gru(x)

        feat = out[:, -1]

        return feat