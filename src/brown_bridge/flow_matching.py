"""Amortize OT conditional paths into a continuous-time vector field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .paths import Geometry, make_time_grid, sample_coupling


class VectorField(Protocol):
    def __call__(self, state: Tensor, time: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class FlowMatchingBatch:
    time: Tensor
    state: Tensor
    conditional_velocity: Tensor


class TimeConditionedMLP(nn.Module):
    """Small vector field for feature-space OT-CFM experiments."""

    def __init__(
        self,
        feature_dim: int,
        *,
        width: int = 512,
        depth: int = 4,
        time_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(time_frequencies, dtype=torch.float32),
            persistent=False,
        )
        input_dim = feature_dim + 1 + 2 * time_frequencies
        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor, time: Tensor) -> Tensor:
        time = time.reshape(-1, 1).to(device=state.device, dtype=state.dtype)
        phase = torch.pi * time * self.frequencies.to(state)
        embedding = torch.cat([time, phase.sin(), phase.cos()], dim=1)
        return self.network(torch.cat([state, embedding], dim=1))


def _spherical_at(
    source: Tensor,
    target: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor]:
    x = F.normalize(source, dim=1)
    y = F.normalize(target, dim=1)
    cosine = (x * y).sum(1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    residual = y - cosine * x
    residual_norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    tangent = residual / residual_norm.clamp_min(1e-8)

    # Exact antipodes have no unique geodesic. Pick a deterministic tangent.
    coordinate = x.abs().argmin(dim=1, keepdim=True)
    basis = torch.zeros_like(x).scatter(1, coordinate, 1.0)
    fallback = F.normalize(basis - (basis * x).sum(1, keepdim=True) * x, dim=1)
    tangent = torch.where(residual_norm > 1e-7, tangent, fallback)

    angle = time[:, None] * theta
    state = torch.cos(angle) * x + torch.sin(angle) * tangent
    velocity = theta * (-torch.sin(angle) * x + torch.cos(angle) * tangent)
    coincident = theta <= 1e-7
    state = torch.where(coincident, x, state)
    velocity = torch.where(coincident, torch.zeros_like(velocity), velocity)
    return state, velocity


def make_ot_cfm_batch(
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    batch_size: int,
    *,
    geometry: Geometry = "spherical",
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> FlowMatchingBatch:
    """Sample the simulation-free OT conditional-flow-matching target."""
    endpoints = sample_coupling(
        plan, source, target, batch_size, generator=generator
    )
    x = endpoints.source.float()
    y = endpoints.target.float()
    time = torch.rand(batch_size, generator=generator)
    if geometry == "ambient":
        state = (1.0 - time[:, None]) * x + time[:, None] * y
        velocity = y - x
    elif geometry == "spherical":
        state, velocity = _spherical_at(x, y, time)
    else:
        raise ValueError(f"unsupported geometry: {geometry}")

    if device is not None:
        time, state, velocity = time.to(device), state.to(device), velocity.to(device)
    return FlowMatchingBatch(time, state, velocity)


def flow_matching_loss(model: VectorField, batch: FlowMatchingBatch) -> Tensor:
    prediction = model(batch.state, batch.time)
    return F.mse_loss(prediction, batch.conditional_velocity)


def _tangent_velocity(state: Tensor, velocity: Tensor) -> Tensor:
    state = F.normalize(state, dim=1)
    return velocity - (velocity * state).sum(1, keepdim=True) * state


@torch.inference_mode()
def integrate_flow(
    model: VectorField,
    initial_state: Tensor,
    *,
    macro_steps: int,
    microsteps_per_macro: int,
    method: Literal["euler", "heun", "rk4"] = "heun",
    geometry: Geometry = "spherical",
) -> tuple[Tensor, Tensor]:
    """Integrate a learned vector field on the requested macro/micro grid."""
    times = make_time_grid(
        macro_steps, microsteps_per_macro, dtype=initial_state.dtype
    ).to(initial_state.device)
    state = initial_state
    if geometry == "spherical":
        state = F.normalize(state, dim=1)
    trajectory = [state.clone()]

    def field(value: Tensor, scalar_time: Tensor) -> Tensor:
        time_batch = scalar_time.expand(len(value))
        velocity = model(value, time_batch)
        return _tangent_velocity(value, velocity) if geometry == "spherical" else velocity

    for left, right in zip(times[:-1], times[1:]):
        step = right - left
        if method == "euler":
            state = state + step * field(state, left)
        elif method == "heun":
            k1 = field(state, left)
            proposal = state + step * k1
            k2 = field(proposal, right)
            state = state + 0.5 * step * (k1 + k2)
        elif method == "rk4":
            midpoint = left + 0.5 * step
            k1 = field(state, left)
            k2 = field(state + 0.5 * step * k1, midpoint)
            k3 = field(state + 0.5 * step * k2, midpoint)
            k4 = field(state + step * k3, right)
            state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        else:
            raise ValueError(f"unsupported integration method: {method}")
        if geometry == "spherical":
            state = F.normalize(state, dim=1)
        trajectory.append(state.clone())
    return times, torch.stack(trajectory)
