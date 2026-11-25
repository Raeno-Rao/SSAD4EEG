"""Spectral score-based anisotropic diffusion for sample-aligned EEG features."""

import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


BANDS = ((1., 4.), (4., 8.), (8., 13.), (13., 30.), (30., 50.))


def band_log_power(x, sampling_rate):
    """Log mean power in physical-Hz bands, averaged over feature channels."""
    spectrum = torch.fft.rfft(x, dim=-1, norm="ortho").abs().square().mean(dim=1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1. / sampling_rate).to(x.device)
    values = []
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        values.append(torch.log(spectrum[:, mask].mean(dim=-1).clamp_min(1e-8)))
    return torch.stack(values, dim=-1)


def spectral_alignment(x, h, sampling_rate):
    # Centering per trial removes scale while preserving relative band structure.
    raw = band_log_power(x, sampling_rate)
    latent = band_log_power(h, sampling_rate)
    raw = raw - raw.mean(dim=-1, keepdim=True)
    latent = latent - latent.mean(dim=-1, keepdim=True)
    return F.mse_loss(latent, raw), raw.detach(), latent.detach()


class ReverseGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class SampleAlignedEncoder(nn.Module):
    def __init__(self, channels, hidden=128):
        super().__init__()
        self.projection = nn.Conv1d(channels, hidden, 1, bias=False)

    def forward(self, x):
        return self.projection(x)


class SpectralGeometry(nn.Module):
    def __init__(self, sampling_rate, eps=1e-3):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.eps = eps
        self.centers = nn.Parameter(torch.tensor([2., 6., 10., 20., 35.]))
        self.log_widths = nn.Parameter(torch.log(torch.tensor([2., 3., 3., 5., 8.])))
        self.logits = nn.Parameter(torch.ones(5))
        self.bias = nn.Parameter(torch.zeros(()))

    def gate(self, length, device):
        freqs = torch.fft.rfftfreq(length, d=1. / self.sampling_rate).to(device)
        centers = self.centers.clamp(0., self.sampling_rate / 2)
        widths = self.log_widths.exp().clamp_min(0.25)
        basis = torch.exp(-0.5 * ((freqs[None] - centers[:, None]) / widths[:, None]).square())
        return torch.sigmoid(self.bias + (self.logits[:, None] * basis).sum(dim=0)).clamp(1e-4, 1. - 1e-4)

    @staticmethod
    def apply(x, response):
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1, norm="ortho") * response, n=x.shape[-1], dim=-1, norm="ortho")

    def responses(self, length, device):
        gate = self.gate(length, device)
        variance = self.eps + 1. - gate.square()
        return gate, variance


class ScoreNet(nn.Module):
    def __init__(self, hidden=128, gate_bins=64):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(gate_bins + 2, hidden), nn.GELU())
        self.condition_low = nn.Linear(hidden, 64)
        self.condition_mid = nn.Linear(hidden, 128)
        self.in_conv = nn.Conv1d(hidden, 64, 3, padding=1)
        self.low = TemporalBlock(64)
        self.down = nn.Conv1d(64, 128, 4, stride=2, padding=1)
        self.middle = TemporalBlock(128)
        self.up = nn.Conv1d(128, 64, 3, padding=1)
        self.high = TemporalBlock(64)
        self.out = nn.Conv1d(64, hidden, 3, padding=1)
        self.gate_bins = gate_bins

    def forward(self, x, t, gate):
        g = F.interpolate(gate[None, None], size=self.gate_bins, mode="linear", align_corners=False).expand(x.shape[0], -1, -1).squeeze(1)
        phase = 2 * math.pi * t
        condition = self.condition(torch.cat((g, torch.sin(phase)[:, None], torch.cos(phase)[:, None]), dim=1))
        skip = self.low(self.in_conv(x) + self.condition_low(condition)[:, :, None])
        middle = self.middle(self.down(skip) + self.condition_mid(condition)[:, :, None])
        up = F.interpolate(middle, size=x.shape[-1], mode="linear", align_corners=False)
        return self.out(self.high(self.up(up) + skip))


class SSAD(nn.Module):
    def __init__(self, channels, classes, subjects, sampling_rate, hidden=128, sigma_min=0.01, sigma_max=10., tau=0.2, steps=4,
                 filtering=True, covariance_mode="learned", refinement="score", noise_enabled=True, gur_mode="learned", subject_alignment=True):
        super().__init__()
        if steps < 1 or covariance_mode not in {"learned", "isotropic", "random", "fixed"}:
            raise ValueError("Invalid steps or covariance mode")
        if refinement not in {"score", "residual", "none"} or gur_mode not in {"learned", "isotropic", "off"}:
            raise ValueError("Invalid refinement or GUR mode")
        self.encoder = SampleAlignedEncoder(channels, hidden)
        self.geometry = SpectralGeometry(sampling_rate)
        self.score = ScoreNet(hidden)
        self.classifier = nn.Linear(hidden, classes)
        self.domain = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, subjects))
        self.sampling_rate = sampling_rate
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.tau = tau
        self.steps = steps
        self.filtering = filtering
        self.covariance_mode = covariance_mode
        self.refinement = refinement
        self.noise_enabled = noise_enabled
        self.gur_mode = gur_mode
        self.subject_alignment = subject_alignment

    @staticmethod
    def trace_mean(response, length):
        weights = torch.full_like(response, 2.)
        weights[0] = 1.
        if length % 2 == 0:
            weights[-1] = 1.
        return (response * weights).sum() / length

    def covariance_response(self, gate, learned, length):
        if self.covariance_mode == "learned":
            return learned
        target_mean = self.trace_mean(learned, length)
        if self.covariance_mode == "isotropic":
            return torch.ones_like(learned) * target_mean
        if self.covariance_mode == "fixed":
            freqs = torch.fft.rfftfreq(length, d=1. / self.sampling_rate).to(gate.device)
            centers = gate.new_tensor([2., 6., 10., 20., 35.])
            widths = gate.new_tensor([2., 3., 3., 5., 8.])
            basis = torch.exp(-0.5 * ((freqs[None] - centers[:, None]) / widths[:, None]).square())
            fixed_gate = torch.sigmoid(basis.mean(dim=0))
            profile = self.geometry.eps + 1. - fixed_gate.square()
        else:
            generator = torch.Generator(device=gate.device).manual_seed(2026)
            random = torch.rand(1, 1, gate.numel(), device=gate.device, generator=generator)
            profile = F.avg_pool1d(F.pad(random, (4, 4), mode="replicate"), 9, stride=1).flatten().clamp_min(1e-3)
        return profile * (target_mean / self.trace_mean(profile, length))

    def variance(self, t):
        return self.sigma_min**2 * ((self.sigma_max / self.sigma_min)**(2 * t) - 1.)

    def variance_derivative(self, t):
        return 2 * math.log(self.sigma_max / self.sigma_min) * (self.sigma_min * (self.sigma_max / self.sigma_min)**t).square()

    def forward(self, x, training_losses=False, deterministic=True, return_details=False):
        h = self.encoder(x)
        gate, learned_covariance = self.geometry.responses(h.shape[-1], h.device)
        covariance = self.covariance_response(gate, learned_covariance, h.shape[-1])
        clean = self.geometry.apply(h, gate) if self.filtering else h
        t = torch.full((x.shape[0],), self.tau, device=x.device)
        if deterministic:
            generator = torch.Generator(device=x.device).manual_seed(42)
            noise = torch.randn(clean.shape, device=x.device, dtype=clean.dtype, generator=generator)
        else:
            noise = torch.randn_like(clean)
        z = clean + self.variance(t).sqrt()[:, None, None] * self.geometry.apply(noise, covariance.sqrt()) if self.noise_enabled else clean
        perturbed = z
        score_fields = []
        dt = (self.tau - 0.001) / self.steps
        if self.refinement != "none":
            for step in range(self.steps):
                current = torch.full_like(t, self.tau - step * dt)
                score = self.score(z, current, gate)
                if return_details:
                    score_fields.append(score)
                if self.refinement == "score":
                    z = z + 0.5 * self.variance_derivative(current)[:, None, None] * self.geometry.apply(score, covariance) * dt
                else:
                    z = z + score
        pooled = z.mean(dim=-1)
        result = {"logits": self.classifier(pooled), "domain_logits": self.domain(ReverseGradient.apply(pooled)), "gate": gate}
        if return_details:
            result.update({"clean": clean, "perturbed": perturbed, "refined": z, "score_fields": score_fields})
        if training_losses:
            if self.refinement == "score":
                random_t = torch.rand_like(t) * 0.999 + 0.001
                eps = torch.randn_like(clean)
                noisy = clean + self.variance(random_t).sqrt()[:, None, None] * self.geometry.apply(eps, covariance.sqrt())
                estimated_score = self.score(noisy, random_t, gate)
                whitened = self.variance(random_t).sqrt()[:, None, None] * self.geometry.apply(estimated_score, covariance.sqrt())
                result["score_loss"] = (whitened + eps).square().mean()
            else:
                result["score_loss"] = z.new_zeros(())
            if self.gur_mode == "off":
                result["gur_loss"] = z.new_zeros(())
            else:
                gur_covariance = covariance if self.gur_mode == "learned" else torch.ones_like(covariance) * self.trace_mean(covariance, h.shape[-1])
                u = 1e-2 + self.variance(t).mean() * gur_covariance
                result["gur_loss"] = 0.5 * (self.trace_mean(u, h.shape[-1]) + z.square().mean() - self.trace_mean(u.log(), h.shape[-1]) - 1.)
            result["align_loss"], result["raw_bands"], result["latent_bands"] = spectral_alignment(x, h, self.sampling_rate)
        return result
