"""High-resolution image latents and reciprocal convolutional bridge fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .continuous_vae import sinusoidal_time_embedding


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return (value + self.main(value)) / math.sqrt(2.0)


class LatentImageAutoencoder(nn.Module):
    """Deterministic 128px autoencoder with an 8x32x32 default latent."""

    def __init__(self, latent_channels: int = 8, base_channels: int = 32) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, 2 * base_channels, 4, 2, 1),
            ResidualBlock(2 * base_channels),
            nn.Conv2d(2 * base_channels, 3 * base_channels, 4, 2, 1),
            ResidualBlock(3 * base_channels),
            nn.Conv2d(3 * base_channels, latent_channels, 3, padding=1),
            nn.Tanh(),
        )
        self.decoder_input = nn.Conv2d(latent_channels, 3 * base_channels, 3, padding=1)
        self.decoder32 = ResidualBlock(3 * base_channels)
        self.decoder64 = nn.Sequential(
            nn.Conv2d(3 * base_channels, 2 * base_channels, 3, padding=1),
            ResidualBlock(2 * base_channels),
        )
        self.decoder128 = nn.Sequential(
            nn.Conv2d(2 * base_channels, base_channels, 3, padding=1),
            ResidualBlock(base_channels),
        )
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, 128, 128):
            raise ValueError("image must have shape [batch, 3, 128, 128]")
        return self.encoder(image)

    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (self.latent_channels, 32, 32):
            raise ValueError(
                f"latent must have shape [batch, {self.latent_channels}, 32, 32]"
            )
        hidden = self.decoder32(self.decoder_input(latent))
        hidden = F.interpolate(hidden, scale_factor=2, mode="bilinear", align_corners=False)
        hidden = self.decoder64(hidden)
        hidden = F.interpolate(hidden, scale_factor=2, mode="bilinear", align_corners=False)
        return self.output(self.decoder128(hidden))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(image)
        return self.decode(latent), latent


class ConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.normalization1 = nn.GroupNorm(min(8, channels), channels, affine=False)
        self.normalization2 = nn.GroupNorm(min(8, channels), channels, affine=False)
        self.convolution1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.convolution2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.affine = nn.Linear(condition_dim, 4 * channels)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        gamma1, beta1, gamma2, beta2 = self.affine(condition).chunk(4, dim=1)
        hidden = (1 + gamma1[:, :, None, None]) * self.normalization1(value)
        hidden = self.convolution1(F.silu(hidden + beta1[:, :, None, None]))
        hidden = (1 + gamma2[:, :, None, None]) * self.normalization2(hidden)
        hidden = self.convolution2(F.silu(hidden + beta2[:, :, None, None]))
        return (value + hidden) / math.sqrt(2.0)


class SpatialReciprocalField(nn.Module):
    """Joint velocity and score field on a spatial image latent."""

    def __init__(
        self,
        latent_channels: int = 8,
        width: int = 64,
        depth: int = 5,
        condition_dim: int = 96,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.condition_dim = condition_dim
        self.condition = nn.Sequential(
            nn.Linear(condition_dim + 1, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.input = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ConditionedResidualBlock(width, condition_dim) for _ in range(depth)]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
            nn.Conv2d(width, 2 * latent_channels, 3, padding=1),
        )

    def forward(self, state: Tensor, time: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
        if state.ndim != 4 or state.shape[1] != self.latent_channels:
            raise ValueError("state must have shape [batch, latent_channels, H, W]")
        embedding = sinusoidal_time_embedding(time, self.condition_dim).to(state)
        embedding = self.condition(torch.cat([embedding, direction[:, None].to(state)], dim=1))
        hidden = self.input(state)
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.output(hidden).chunk(2, dim=1)


@dataclass(frozen=True)
class SpatialReciprocalBatch:
    state: Tensor
    time: Tensor
    direction: Tensor
    conditional_velocity: Tensor
    scaled_score: Tensor
    gamma: Tensor
    flow_weight: Tensor


def make_spatial_reciprocal_batch(
    source: Tensor,
    target: Tensor,
    *,
    diffusivity: float,
    generator: torch.Generator | None = None,
) -> SpatialReciprocalBatch:
    """Sample both orientations of endpoint-conditioned Brownian bridges."""
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("source and target must have a common [batch, C, H, W] shape")
    if diffusivity <= 0:
        raise ValueError("diffusivity must be positive")
    batch = len(source)
    direction = 2 * torch.randint(0, 2, (batch,), generator=generator) - 1
    forward = direction > 0
    select = forward.reshape(-1, 1, 1, 1)
    left = torch.where(select, source, target)
    right = torch.where(select, target, source)
    time = 0.01 + 0.98 * torch.rand(batch, generator=generator)
    noise = torch.randn(source.shape, generator=generator, dtype=source.dtype)
    gamma = torch.sqrt(2.0 * diffusivity * time * (1.0 - time)).clamp_min(1e-6)
    shape = (-1, 1, 1, 1)
    mean = (1.0 - time.reshape(shape)) * left + time.reshape(shape) * right
    state = mean + gamma.reshape(shape) * noise
    gamma_dot = diffusivity * (1.0 - 2.0 * time) / gamma
    velocity = right - left + gamma_dot.reshape(shape) * noise
    weight = gamma.square() / (gamma.square() + 0.0004)
    return SpatialReciprocalBatch(
        state,
        time,
        direction,
        velocity,
        -noise,
        gamma,
        weight,
    )


@torch.no_grad()
def integrate_spatial_probability_flow(
    field: SpatialReciprocalField,
    initial: Tensor,
    *,
    direction: int = 1,
    steps: int = 24,
) -> Tensor:
    """Heun integration of the learned marginal current velocity."""
    if direction not in (-1, 1) or steps < 1:
        raise ValueError("direction must be +/-1 and steps must be positive")
    state = initial
    dt = 1.0 / steps
    sign = torch.full((len(state),), float(direction), device=state.device)
    trajectory = [state]
    for index in range(steps):
        left = torch.full((len(state),), index * dt, device=state.device)
        right = left + dt
        velocity, _ = field(state, left, sign)
        proposal = state + dt * velocity
        corrected, _ = field(proposal, right, sign)
        state = state + 0.5 * dt * (velocity + corrected)
        trajectory.append(state)
    return torch.stack(trajectory)


@torch.no_grad()
def integrate_spatial_sde(
    field: SpatialReciprocalField,
    initial: Tensor,
    *,
    diffusivity: float,
    direction: int = 1,
    steps: int = 24,
    noise: Tensor | None = None,
) -> Tensor:
    """Euler integration of current velocity plus score and Brownian noise."""
    if diffusivity < 0 or direction not in (-1, 1) or steps < 1:
        raise ValueError("invalid diffusivity, direction, or step count")
    state = initial
    dt = 1.0 / steps
    sign = torch.full((len(state),), float(direction), device=state.device)
    if noise is None:
        noise = torch.randn((steps, *initial.shape), device=initial.device)
    if tuple(noise.shape) != (steps, *initial.shape):
        raise ValueError("noise has the wrong shape")
    trajectory = [state]
    for index in range(steps):
        time = torch.full((len(state),), index * dt, device=state.device)
        velocity, score = field(state, time, sign)
        state = (
            state
            + dt * (velocity + diffusivity * score)
            + math.sqrt(2.0 * diffusivity * dt) * noise[index]
        )
        trajectory.append(state)
    return torch.stack(trajectory)
