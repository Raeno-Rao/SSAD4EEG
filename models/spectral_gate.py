#models/spectral_gate.py
import math

import torch
import torch.nn as nn


class SpectralGate(nn.Module):
    """
    Learnable EEG frequency gate.

    g(f) in (0,1)
    """

    def __init__(
        self,
        seq_len=1000,
        sampling_rate=250,
        num_bands=5,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.freq_bins = seq_len // 2 + 1

        freqs = torch.linspace(0, sampling_rate / 2, self.freq_bins)
        self.register_buffer("freqs", freqs)

        centers = torch.tensor([2.0, 6.0, 10.5, 21.5, 40.0])
        widths = torch.tensor([2.0, 2.0, 3.0, 5.0, 8.0])

        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

        self.phi = nn.Parameter(torch.zeros(num_bands))

    def build_gate(self):
        basis = []

        for c, w in zip(self.centers, self.widths):
            psi = torch.exp(-(self.freqs - c) ** 2 / (2 * w**2))
            basis.append(psi)

        basis = torch.stack(basis, dim=0)

        weights = torch.softmax(self.phi, dim=0)

        gate = torch.sum(weights[:, None] * basis, dim=0)
        gate = torch.sigmoid(5 * gate)

        return gate

    def forward(self, x):
        """
        x: (B, C, T)
        """

        gate = self.build_gate()

        x_fft = torch.fft.rfft(x, dim=-1)

        gate = gate[None, None, :]

        x_fft = x_fft * gate

        x_out = torch.fft.irfft(x_fft, n=x.shape[-1], dim=-1)

        return x_out, gate.squeeze()