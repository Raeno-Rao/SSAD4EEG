#models/faib_net.py
import torch
import torch.nn as nn

from models.encoder import EEGEncoder
from models.spectral_gate import SpectralGate
from models.stochastic_ib import StochasticIB


class FAIBNet(nn.Module):
    def __init__(
        self,
        in_channels=22,
        seq_len=1000,
        hidden_dim=64,
        num_classes=4,
        num_bands=5,
        noise_scale=0.1,
    ):
        super().__init__()

        self.gate = SpectralGate(
            seq_len=seq_len,
            num_bands=num_bands,
        )

        self.ib = StochasticIB(noise_scale=noise_scale)

        self.encoder = EEGEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )

        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, gate = self.gate(x)

        z, kl = self.ib(x, gate)

        feat = self.encoder(z)

        logits = self.cls(feat)

        return {
            "logits": logits,
            "kl": kl,
            "gate": gate,
        }