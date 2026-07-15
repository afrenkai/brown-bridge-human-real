"""Compact continuous-time VAE observation model for feature-space bridges."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_time_embedding(time: Tensor, dimension: int) -> Tensor:
    if dimension % 2:
        raise ValueError("time embedding dimension must be even")
    half = dimension // 2
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), half, device=time.device)
    )
    angles = time.float().reshape(-1, 1) * frequencies.reshape(1, -1) * 2.0 * math.pi
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class DownBlock(nn.Module):
    def __init__(self, source: int, target: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(source, target, 4, 2, 1),
            nn.GroupNorm(min(8, target), target),
            nn.SiLU(),
            nn.Conv2d(target, target, 3, 1, 1),
            nn.GroupNorm(min(8, target), target),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.main(value)


class ConditionedUpBlock(nn.Module):
    def __init__(self, source: int, target: int, condition_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(source, target, 3, 1, 1)
        self.conv2 = nn.Conv2d(target, target, 3, 1, 1)
        self.norm1 = nn.GroupNorm(min(8, target), target, affine=False)
        self.norm2 = nn.GroupNorm(min(8, target), target, affine=False)
        self.affine = nn.Linear(condition_dim, 4 * target)
        self.skip = nn.Conv2d(source, target, 1)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        affine = self.affine(condition).reshape(len(value), -1, 1, 1)
        gamma1, beta1, gamma2, beta2 = affine.chunk(4, dim=1)
        hidden = self.conv1(value)
        hidden = F.silu((1.0 + gamma1) * self.norm1(hidden) + beta1)
        hidden = self.conv2(hidden)
        hidden = F.silu((1.0 + gamma2) * self.norm2(hidden) + beta2)
        return (hidden + self.skip(value)) / math.sqrt(2.0)


class ContinuousConditionalVAE(nn.Module):
    """Decode a spatial latent under DINO global/patch state and bridge time."""

    def __init__(
        self,
        *,
        image_size: int = 128,
        global_dim: int = 192,
        patch_dim: int = 192,
        patch_grid: int = 4,
        latent_grid: int = 4,
        latent_channels: int = 24,
        condition_dim: int = 128,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if image_size != 128:
            raise ValueError("this compact architecture currently supports 128x128 images")
        if latent_grid not in (4, 8, 16):
            raise ValueError("latent_grid must be 4, 8, or 16")
        self.image_size = image_size
        self.patch_grid = patch_grid
        self.latent_grid = latent_grid
        self.latent_channels = latent_channels
        self.conditioning_scale = 1.0
        down_steps = int(math.log2(image_size // latent_grid))
        channels = [min((index + 1) * base_channels, 5 * base_channels) for index in range(down_steps)]
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(3, channels[0], 3, 1, 1),
            nn.SiLU(),
        ]
        for index in range(down_steps):
            source = channels[max(0, index - 1)]
            target = channels[index]
            encoder_layers.append(DownBlock(source, target))
        self.encoder = nn.Sequential(*encoder_layers)
        self.posterior = nn.Conv2d(channels[-1], 2 * latent_channels, 1)
        self.global_condition = nn.Sequential(
            nn.Linear(global_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.time_condition = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        patch_channels = 48
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, patch_channels),
            nn.SiLU(),
        )
        self.decoder_input = nn.Conv2d(latent_channels + patch_channels, channels[-1], 3, 1, 1)
        decoder_blocks: list[nn.Module] = []
        source_channel = channels[-1]
        for target_channel in reversed(channels):
            decoder_blocks.append(
                ConditionedUpBlock(source_channel, target_channel, condition_dim)
            )
            source_channel = target_channel
        self.decoder = nn.ModuleList(decoder_blocks)
        self.output = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channels[0], 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 4 or image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"image must have shape [batch, 3, {self.image_size}, {self.image_size}]")
        mean, log_variance = self.posterior(self.encoder(image)).chunk(2, dim=1)
        return mean, log_variance.clamp(-8.0, 6.0)

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def decode(
        self,
        latent: Tensor,
        time: Tensor,
        global_state: Tensor,
        patch_state: Tensor,
    ) -> Tensor:
        expected_patches = self.patch_grid * self.patch_grid
        if patch_state.ndim != 3 or patch_state.shape[1] != expected_patches:
            raise ValueError(f"patch_state must have {expected_patches} tokens")
        if latent.shape[-2:] != (self.latent_grid, self.latent_grid):
            raise ValueError("latent has the wrong spatial grid")
        condition = self.global_condition(F.normalize(global_state.float(), dim=1))
        condition = condition + self.time_condition(
            sinusoidal_time_embedding(time, condition.shape[1]).to(condition)
        )
        condition = self.conditioning_scale * condition
        patches = self.patch_projection(F.normalize(patch_state.float(), dim=2))
        patches = self.conditioning_scale * patches
        patches = patches.transpose(1, 2).reshape(
            len(latent), -1, self.patch_grid, self.patch_grid
        )
        if self.latent_grid != self.patch_grid:
            patches = F.interpolate(
                patches,
                size=(self.latent_grid, self.latent_grid),
                mode="bilinear",
                align_corners=False,
            )
        hidden = self.decoder_input(torch.cat([latent, patches.to(latent)], dim=1))
        for block in self.decoder:
            hidden = block(hidden, condition)
        return self.output(hidden)

    def forward(
        self,
        image: Tensor,
        time: Tensor,
        global_state: Tensor,
        patch_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(image)
        reconstruction = self.decode(
            self.reparameterize(mean, log_variance), time, global_state, patch_state
        )
        return reconstruction, mean, log_variance


def diagonal_gaussian_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
    return 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance).mean()


def latent_brownian_bridge(
    source: Tensor,
    target: Tensor,
    time: Tensor,
    *,
    diffusivity: float = 0.0,
    noise: Tensor | None = None,
) -> Tensor:
    if source.shape != target.shape:
        raise ValueError("latent endpoints must have equal shapes")
    weight = time.to(source).reshape(-1, 1, 1, 1)
    mean = torch.lerp(source, target, weight)
    if diffusivity == 0:
        return mean
    if diffusivity < 0:
        raise ValueError("diffusivity must be nonnegative")
    if noise is None:
        noise = torch.randn_like(mean)
    scale = torch.sqrt(2.0 * diffusivity * weight * (1.0 - weight))
    return mean + scale * noise
