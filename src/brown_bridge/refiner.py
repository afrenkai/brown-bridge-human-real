"""Small image decoder conditioned on intermediate reciprocal-SDE states."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvNormAct(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, stride, 1),
            nn.GroupNorm(min(8, dim_out), dim_out),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.main(value)


class ConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels, affine=False)
        self.norm2 = nn.GroupNorm(8, channels, affine=False)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.affine = nn.Linear(condition_dim, 4 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        affine = self.affine(condition).view(len(value), -1, 1, 1)
        gamma1, beta1, gamma2, beta2 = affine.chunk(4, dim=1)
        hidden = (1.0 + gamma1) * self.norm1(value) + beta1
        hidden = self.conv1(F.silu(hidden))
        hidden = (1.0 + gamma2) * self.norm2(hidden) + beta2
        return value + self.conv2(F.silu(hidden)) / math.sqrt(2.0)


class SDEConditionedRefiner(nn.Module):
    """Warp and recolor a base render using ``(z0, zt, z1, t)``.

    The predicted deformation is multiplied by ``t(1-t)``. Consequently the
    module is exactly the identity map at both bridge endpoints for any model
    parameters.
    """

    def __init__(
        self,
        feature_dim: int = 192,
        *,
        condition_dim: int = 128,
        max_flow: float = 0.08,
        max_residual: float = 0.12,
        output_grid_size: int | None = None,
    ) -> None:
        super().__init__()
        if output_grid_size is not None and output_grid_size < 2:
            raise ValueError("output_grid_size must be at least two")
        self.max_flow = max_flow
        self.max_residual = max_residual
        self.output_grid_size = output_grid_size
        self.condition = nn.Sequential(
            nn.Linear(3 * feature_dim + 3, 256),
            nn.SiLU(),
            nn.Linear(256, condition_dim),
            nn.SiLU(),
        )
        self.down1 = ConvNormAct(6, 32, stride=2)
        self.down2 = ConvNormAct(32, 64, stride=2)
        self.down3 = ConvNormAct(64, 96, stride=2)
        self.middle1 = ConditionedResidualBlock(96, condition_dim)
        self.middle2 = ConditionedResidualBlock(96, condition_dim)
        self.up2 = ConvNormAct(96 + 64, 64)
        self.up1 = ConvNormAct(64 + 32, 32)
        self.output = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 5, 3, 1, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def time_embedding(self, time: Tensor) -> Tensor:
        value = time.reshape(-1, 1)
        return torch.cat([value, torch.sin(torch.pi * value), torch.cos(torch.pi * value)], dim=1)

    def forward(
        self,
        base_image: Tensor,
        source_image: Tensor,
        source_state: Tensor,
        bridge_state: Tensor,
        target_state: Tensor,
        time: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if base_image.shape != source_image.shape or base_image.ndim != 4:
            raise ValueError("base/source images must have a common NCHW shape")
        if source_state.shape != bridge_state.shape or source_state.shape != target_state.shape:
            raise ValueError("all bridge states must have a common [batch, features] shape")
        if len(time.reshape(-1)) != len(base_image):
            raise ValueError("time batch must match image batch")
        condition = self.condition(
            torch.cat(
                [source_state, bridge_state, target_state, self.time_embedding(time)],
                dim=1,
            )
        )
        first = self.down1(torch.cat([base_image, source_image], dim=1))
        second = self.down2(first)
        hidden = self.down3(second)
        hidden = self.middle2(self.middle1(hidden, condition), condition)
        hidden = F.interpolate(hidden, size=second.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.up2(torch.cat([hidden, second], dim=1))
        hidden = F.interpolate(hidden, size=first.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.up1(torch.cat([hidden, first], dim=1))
        hidden = F.interpolate(hidden, size=base_image.shape[-2:], mode="bilinear", align_corners=False)
        raw_flow, raw_residual = self.output(hidden).split((2, 3), dim=1)
        if self.output_grid_size is not None:
            low_size = (self.output_grid_size, self.output_grid_size)

            def band_limit(value: Tensor) -> Tensor:
                low = F.adaptive_avg_pool2d(value, low_size)
                return F.interpolate(
                    low,
                    size=base_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            raw_flow = band_limit(raw_flow)
            raw_residual = band_limit(raw_residual)
        endpoint_weight = (time * (1.0 - time)).reshape(-1, 1, 1, 1).to(base_image)
        flow = endpoint_weight * self.max_flow * torch.tanh(raw_flow)
        residual = endpoint_weight * self.max_residual * torch.tanh(raw_residual)
        height, width = base_image.shape[-2:]
        vertical, horizontal = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=base_image.device, dtype=base_image.dtype),
            torch.linspace(-1.0, 1.0, width, device=base_image.device, dtype=base_image.dtype),
            indexing="ij",
        )
        grid = torch.stack([horizontal, vertical], dim=-1)[None].expand(len(base_image), -1, -1, -1)
        warped = F.grid_sample(
            base_image,
            grid + flow.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        modified = (warped + residual).clamp(-1.0, 1.0)
        output = torch.where(endpoint_weight == 0, base_image, modified)
        return output, flow, residual


def deformation_temporal_loss(flow: Tensor, residual: Tensor) -> Tensor:
    """Penalize deformation magnitude and discrete temporal acceleration."""
    if flow.ndim != 5 or residual.ndim != 5 or flow.shape[:2] != residual.shape[:2]:
        raise ValueError("deformations must have shape [batch, time, channels, H, W]")
    if flow.shape[1] < 3:
        raise ValueError("at least three time points are required")

    def curvature(value: Tensor) -> Tensor:
        acceleration = value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]
        return acceleration.square().mean()

    return curvature(flow) + curvature(residual) + 0.05 * (
        flow.square().mean() + residual.square().mean()
    )
