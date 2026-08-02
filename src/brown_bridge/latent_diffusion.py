"""Bidirectional conditional diffusion kernels on spatial image latents."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .residual_diffusion import TimeResidualBlock, sinusoidal_embedding


class BidirectionalLatentDenoiser(nn.Module):
    """One conditional U-Net for source-to-target sampling in either direction."""

    def __init__(
        self,
        latent_channels: int = 8,
        base_channels: int = 48,
        time_dim: int = 128,
        latent_size: int = 32,
        transformer_blocks: int = 0,
        transformer_heads: int = 6,
        descriptor_dim: int = 0,
    ) -> None:
        super().__init__()
        if latent_size % 4:
            raise ValueError("latent_size must be divisible by four")
        if descriptor_dim < 0:
            raise ValueError("descriptor_dim must be nonnegative")
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.time_dim = time_dim
        self.descriptor_dim = descriptor_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim + 1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.descriptor_mlp = (
            nn.Sequential(
                nn.LayerNorm(descriptor_dim),
                nn.Linear(descriptor_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            if descriptor_dim
            else None
        )
        self.input = nn.Conv2d(2 * latent_channels, base_channels, 3, padding=1)
        self.down1 = TimeResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, 2 * base_channels, 4, 2, 1)
        self.down2 = TimeResidualBlock(2 * base_channels, 2 * base_channels, time_dim)
        self.downsample2 = nn.Conv2d(2 * base_channels, 3 * base_channels, 4, 2, 1)
        self.middle1 = TimeResidualBlock(3 * base_channels, 3 * base_channels, time_dim)
        bottleneck_channels = 3 * base_channels
        if transformer_blocks:
            if bottleneck_channels % transformer_heads:
                raise ValueError("bottleneck channels must divide transformer heads")
            layer = nn.TransformerEncoderLayer(
                d_model=bottleneck_channels,
                nhead=transformer_heads,
                dim_feedforward=4 * bottleneck_channels,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, transformer_blocks)
            bottleneck_tokens = (latent_size // 4) ** 2
            self.position = nn.Parameter(
                torch.randn(1, bottleneck_tokens, bottleneck_channels) * 0.02
            )
        else:
            self.transformer = nn.Identity()
            self.register_parameter("position", None)
        self.middle2 = TimeResidualBlock(3 * base_channels, 3 * base_channels, time_dim)
        self.upsample2 = nn.ConvTranspose2d(
            3 * base_channels, 2 * base_channels, 4, 2, 1
        )
        self.up2 = TimeResidualBlock(4 * base_channels, 2 * base_channels, time_dim)
        self.upsample1 = nn.ConvTranspose2d(
            2 * base_channels, base_channels, 4, 2, 1
        )
        self.up1 = TimeResidualBlock(2 * base_channels, base_channels, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, 3, padding=1),
        )

    def forward(
        self,
        noisy_target: Tensor,
        condition: Tensor,
        time: Tensor,
        direction: Tensor,
        descriptor: Tensor | None = None,
    ) -> Tensor:
        if (
            noisy_target.shape != condition.shape
            or noisy_target.ndim != 4
            or noisy_target.shape[1] != self.latent_channels
        ):
            raise ValueError("target and condition must share a spatial latent shape")
        if time.numel() != len(noisy_target) or direction.numel() != len(noisy_target):
            raise ValueError("time and direction must have one value per latent")
        embedding = sinusoidal_embedding(time.reshape(-1).to(noisy_target), self.time_dim)
        embedding = self.time_mlp(
            torch.cat([embedding, direction.reshape(-1, 1).to(noisy_target)], dim=1)
        )
        if self.descriptor_mlp is not None:
            if descriptor is None or descriptor.shape != (
                len(noisy_target), self.descriptor_dim
            ):
                raise ValueError(
                    f"descriptor must have shape [batch, {self.descriptor_dim}]"
                )
            embedding = embedding + self.descriptor_mlp(descriptor.to(noisy_target))
        elif descriptor is not None:
            raise ValueError("this denoiser was configured without descriptors")
        level1 = self.down1(
            self.input(torch.cat([noisy_target, condition], dim=1)), embedding
        )
        level2 = self.down2(self.downsample1(level1), embedding)
        hidden = self.middle1(self.downsample2(level2), embedding)
        tokens = hidden.flatten(2).transpose(1, 2)
        if self.position is not None:
            if tokens.shape[1] != self.position.shape[1]:
                raise ValueError(
                    "input spatial size does not match the configured latent_size"
                )
            tokens = tokens + self.position.to(tokens)
        tokens = self.transformer(tokens)
        hidden = self.middle2(tokens.transpose(1, 2).reshape_as(hidden), embedding)
        hidden = self.up2(
            torch.cat([self.upsample2(hidden), level2], dim=1), embedding
        )
        hidden = self.up1(
            torch.cat([self.upsample1(hidden), level1], dim=1), embedding
        )
        return self.output(hidden)


def q_sample(clean: Tensor, noise: Tensor, time_index: Tensor, alpha_bar: Tensor) -> Tensor:
    """Sample the variance-preserving forward diffusion at integer times."""
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")
    cumulative = alpha_bar[time_index].to(clean)
    shape = (len(clean),) + (1,) * (clean.ndim - 1)
    return (
        cumulative.sqrt().reshape(shape) * clean
        + (1.0 - cumulative).sqrt().reshape(shape) * noise
    )


@torch.no_grad()
def conditional_latent_ddim_sample(
    model: BidirectionalLatentDenoiser,
    condition: Tensor,
    alpha_bar: Tensor,
    *,
    direction: int,
    sampling_steps: int = 20,
    initial_noise: Tensor | None = None,
    clean_clip: float | None = 1.0,
    start_fraction: float = 1.0,
    guidance_scale: float = 1.0,
    initial_clean: Tensor | None = None,
    descriptor: Tensor | None = None,
) -> Tensor:
    """Sample one endpoint from the forward or backward conditional kernel."""
    if direction not in (-1, 1):
        raise ValueError("direction must be +/-1")
    total_steps = len(alpha_bar)
    if not 0.0 < start_fraction <= 1.0:
        raise ValueError("start_fraction must lie in (0, 1]")
    start_index = max(1, round((total_steps - 1) * start_fraction))
    if not 2 <= sampling_steps <= start_index + 1:
        raise ValueError("sampling_steps must fit within the truncated schedule")
    if clean_clip is not None and clean_clip <= 0:
        raise ValueError("clean_clip must be positive")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be nonnegative")
    if descriptor is not None and len(descriptor) != len(condition):
        raise ValueError("descriptor batch must match the condition")
    noise = torch.randn_like(condition) if initial_noise is None else initial_noise.clone()
    if start_fraction < 1.0:
        clean_start = condition if initial_clean is None else initial_clean
        if clean_start.shape != condition.shape:
            raise ValueError("initial_clean must match the condition shape")
        start_alpha = alpha_bar[start_index].to(condition)
        value = start_alpha.sqrt() * clean_start + (1.0 - start_alpha).sqrt() * noise
    else:
        value = noise
    indices = torch.linspace(
        start_index, 0, sampling_steps, device=condition.device
    ).round().long()
    direction_batch = torch.full(
        (len(condition),), float(direction), device=condition.device
    )
    for position, index in enumerate(indices):
        current = alpha_bar[index].to(condition)
        time = torch.full(
            (len(condition),),
            index.item() / max(total_steps - 1, 1),
            device=condition.device,
        )
        predicted_noise = model(
            value, condition, time, direction_batch, descriptor=descriptor
        )
        if guidance_scale != 1.0:
            unconditional_noise = model(
                value,
                torch.zeros_like(condition),
                time,
                direction_batch,
                descriptor=(
                    torch.zeros_like(descriptor) if descriptor is not None else None
                ),
            )
            predicted_noise = unconditional_noise + guidance_scale * (
                predicted_noise - unconditional_noise
            )
        clean = (value - (1.0 - current).sqrt() * predicted_noise) / current.sqrt()
        if clean_clip is not None:
            clean = clean.clamp(-clean_clip, clean_clip)
        if position == len(indices) - 1:
            value = clean
            continue
        following = alpha_bar[indices[position + 1]].to(condition)
        value = following.sqrt() * clean + (1.0 - following).sqrt() * predicted_noise
    return value
