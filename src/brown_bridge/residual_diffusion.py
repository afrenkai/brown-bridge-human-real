"""A compact conditional diffusion decoder for VAE reconstruction residuals."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def cosine_alpha_bar(steps: int, *, device: torch.device | str | None = None) -> Tensor:
    """Nichol--Dhariwal cosine cumulative noise schedule, indexed from 0 to T-1."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    time = torch.linspace(0.0, 1.0, steps + 1, device=device)
    cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    return cumulative[1:].clamp(1e-5, 1.0 - 1e-5)


def sinusoidal_embedding(time: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequencies[None] * 1000.0
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(embedding, (0, dimension - embedding.shape[1]))


class TimeResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, time_dim: int) -> None:
        super().__init__()
        self.normalization1 = nn.GroupNorm(min(8, input_channels), input_channels)
        self.convolution1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, 2 * output_channels)
        self.normalization2 = nn.GroupNorm(min(8, output_channels), output_channels)
        self.convolution2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, value: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.convolution1(F.silu(self.normalization1(value)))
        scale, shift = self.time_projection(time_embedding).chunk(2, dim=1)
        hidden = self.normalization2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.convolution2(F.silu(hidden))
        return hidden + self.skip(value)


class ResidualDenoiser(nn.Module):
    """Small U-Net predicting noise from a noisy residual and VAE base image."""

    def __init__(self, base_channels: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.input = nn.Conv2d(6, base_channels, 3, padding=1)
        self.down1 = TimeResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, 2 * base_channels, 4, 2, 1)
        self.down2 = TimeResidualBlock(2 * base_channels, 2 * base_channels, time_dim)
        self.downsample2 = nn.Conv2d(2 * base_channels, 3 * base_channels, 4, 2, 1)
        self.middle1 = TimeResidualBlock(3 * base_channels, 3 * base_channels, time_dim)
        self.middle2 = TimeResidualBlock(3 * base_channels, 3 * base_channels, time_dim)
        self.upsample2 = nn.ConvTranspose2d(3 * base_channels, 2 * base_channels, 4, 2, 1)
        self.up2 = TimeResidualBlock(4 * base_channels, 2 * base_channels, time_dim)
        self.upsample1 = nn.ConvTranspose2d(2 * base_channels, base_channels, 4, 2, 1)
        self.up1 = TimeResidualBlock(2 * base_channels, base_channels, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def forward(self, noisy_residual: Tensor, condition: Tensor, time: Tensor) -> Tensor:
        if noisy_residual.shape != condition.shape or noisy_residual.shape[1] != 3:
            raise ValueError("noisy_residual and condition must have shape [batch, 3, H, W]")
        embedding = self.time_mlp(sinusoidal_embedding(time.float(), self.time_dim))
        level1 = self.down1(self.input(torch.cat([noisy_residual, condition], dim=1)), embedding)
        level2 = self.down2(self.downsample1(level1), embedding)
        hidden = self.middle2(self.middle1(self.downsample2(level2), embedding), embedding)
        hidden = self.up2(torch.cat([self.upsample2(hidden), level2], dim=1), embedding)
        hidden = self.up1(torch.cat([self.upsample1(hidden), level1], dim=1), embedding)
        return self.output(hidden)


def q_sample(clean: Tensor, noise: Tensor, time_index: Tensor, alpha_bar: Tensor) -> Tensor:
    """Sample q(x_t | x_0) for a batch of integer diffusion indices."""
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")
    cumulative = alpha_bar[time_index].to(clean)
    shape = (len(clean),) + (1,) * (clean.ndim - 1)
    return cumulative.sqrt().reshape(shape) * clean + (1.0 - cumulative).sqrt().reshape(shape) * noise


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    condition: Tensor,
    alpha_bar: Tensor,
    *,
    sampling_steps: int = 20,
    initial_noise: Tensor | None = None,
    start_fraction: float = 1.0,
) -> Tensor:
    """Deterministic DDIM sampling from a zero-residual prior at a chosen noise time."""
    total_steps = len(alpha_bar)
    if not 1 <= sampling_steps <= total_steps:
        raise ValueError("sampling_steps must be between 1 and the schedule length")
    if not 0.0 < start_fraction <= 1.0:
        raise ValueError("start_fraction must be in (0, 1]")
    start_index = max(1, round((total_steps - 1) * start_fraction))
    noise = torch.randn_like(condition) if initial_noise is None else initial_noise.clone()
    # q(x_tau | x_0=0): truncated diffusion refines the condition instead of
    # asking the small decoder to synthesize an entire residual from pure noise.
    value = (1.0 - alpha_bar[start_index].to(condition)).sqrt() * noise
    indices = torch.linspace(start_index, 0, sampling_steps, device=condition.device).long()
    for position, index in enumerate(indices):
        current = alpha_bar[index].to(condition)
        time = torch.full(
            (len(condition),), index.item() / max(total_steps - 1, 1), device=condition.device
        )
        predicted_noise = model(value, condition, time)
        clean = (value - (1.0 - current).sqrt() * predicted_noise) / current.sqrt()
        clean = clean.clamp(-1.5, 1.5)
        if position == len(indices) - 1:
            value = clean
            continue
        following = alpha_bar[indices[position + 1]].to(condition)
        value = following.sqrt() * clean + (1.0 - following).sqrt() * predicted_noise
    return value


def refine_image(base: Tensor, normalized_residual: Tensor, *, residual_scale: float = 0.25) -> Tensor:
    return (base + residual_scale * normalized_residual).clamp(0.0, 1.0)
