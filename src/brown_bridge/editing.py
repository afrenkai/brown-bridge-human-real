"""Source-anchored feature editing and a generic FlowEdit integrator.

Residual feature matching learns corrections around a particular DINO feature.
FlowEdit is different: it requires two already-trained conditional generative
velocity fields and combines their velocity difference at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .paths import Geometry, make_time_grid, sample_coupling


class ResidualVectorField(Protocol):
    def __call__(self, residual: Tensor, time: Tensor, anchor: Tensor) -> Tensor: ...


class ConditionalGenerativeField(Protocol):
    def __call__(self, state: Tensor, time: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class ResidualEditBatch:
    anchor: Tensor
    residual: Tensor
    time: Tensor
    conditional_velocity: Tensor


class SourceConditionedResidualMLP(nn.Module):
    """A source-anchored correction field, not a global distribution flow."""

    def __init__(
        self,
        feature_dim: int,
        *,
        width: int = 512,
        depth: int = 4,
        time_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(time_frequencies, dtype=torch.float32),
            persistent=False,
        )
        input_dim = 2 * feature_dim + 1 + 2 * time_frequencies
        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, residual: Tensor, time: Tensor, anchor: Tensor) -> Tensor:
        time = time.reshape(-1, 1).to(residual)
        phase = torch.pi * time * self.frequencies.to(residual)
        time_embedding = torch.cat([time, phase.sin(), phase.cos()], dim=1)
        return self.network(torch.cat([residual, anchor, time_embedding], dim=1))


def _fallback_tangent(anchor: Tensor) -> Tensor:
    coordinate = anchor.abs().argmin(dim=1, keepdim=True)
    basis = torch.zeros_like(anchor).scatter(1, coordinate, 1.0)
    return F.normalize(basis - (basis * anchor).sum(1, keepdim=True) * anchor, dim=1)


def sphere_log_map(anchor: Tensor, target: Tensor) -> Tensor:
    """Map a spherical target into the tangent space at ``anchor``."""
    x = F.normalize(anchor, dim=1)
    y = F.normalize(target, dim=1)
    cosine = (x * y).sum(1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    direction = y - cosine * x
    norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    unit = direction / norm.clamp_min(1e-12)
    unit = torch.where(norm > 1e-7, unit, _fallback_tangent(x))
    return torch.where(theta > 1e-7, theta * unit, torch.zeros_like(unit))


def sphere_exp_map(anchor: Tensor, tangent: Tensor) -> Tensor:
    """Decode a tangent residual into a normalized edited feature."""
    x = F.normalize(anchor, dim=1)
    tangent = tangent - (tangent * x).sum(1, keepdim=True) * x
    norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    unit = tangent / norm.clamp_min(1e-12)
    result = torch.cos(norm) * x + torch.sin(norm) * unit
    return torch.where(norm > 1e-7, result, x)


def make_residual_edit_batch(
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    batch_size: int,
    *,
    geometry: Geometry = "spherical",
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> ResidualEditBatch:
    """Use OT only to propose training pairs, then learn source-local edits."""
    endpoints = sample_coupling(
        plan, source, target, batch_size, generator=generator
    )
    anchor = endpoints.source.float()
    if geometry == "ambient":
        final_residual = endpoints.target.float() - anchor
    elif geometry == "spherical":
        anchor = F.normalize(anchor, dim=1)
        final_residual = sphere_log_map(anchor, endpoints.target.float())
    else:
        raise ValueError(f"unsupported geometry: {geometry}")
    time = torch.rand(batch_size, generator=generator)
    residual = time[:, None] * final_residual
    if device is not None:
        anchor = anchor.to(device)
        residual = residual.to(device)
        time = time.to(device)
        final_residual = final_residual.to(device)
    return ResidualEditBatch(anchor, residual, time, final_residual)


def residual_flow_matching_loss(
    model: ResidualVectorField,
    batch: ResidualEditBatch,
) -> Tensor:
    prediction = model(batch.residual, batch.time, batch.anchor)
    return F.mse_loss(prediction, batch.conditional_velocity)


@torch.inference_mode()
def integrate_residual_edit(
    model: ResidualVectorField,
    source: Tensor,
    *,
    macro_steps: int,
    microsteps_per_macro: int,
    geometry: Geometry = "spherical",
    method: Literal["euler", "heun"] = "heun",
) -> tuple[Tensor, Tensor, Tensor]:
    """Integrate source-conditioned residuals and decode edited features."""
    times = make_time_grid(
        macro_steps, microsteps_per_macro, dtype=source.dtype
    ).to(source.device)
    anchor = F.normalize(source, dim=1) if geometry == "spherical" else source
    residual = torch.zeros_like(anchor)
    residual_path = [residual.clone()]
    feature_path = [anchor.clone()]

    def field(value: Tensor, time: Tensor) -> Tensor:
        velocity = model(value, time.expand(len(value)), anchor)
        if geometry == "spherical":
            velocity = velocity - (velocity * anchor).sum(1, keepdim=True) * anchor
        return velocity

    for left, right in zip(times[:-1], times[1:]):
        step = right - left
        first = field(residual, left)
        if method == "euler":
            residual = residual + step * first
        elif method == "heun":
            proposal = residual + step * first
            residual = residual + 0.5 * step * (first + field(proposal, right))
        else:
            raise ValueError(f"unsupported integration method: {method}")
        if geometry == "spherical":
            residual = residual - (residual * anchor).sum(1, keepdim=True) * anchor
            edited = sphere_exp_map(anchor, residual)
        else:
            edited = anchor + residual
        residual_path.append(residual.clone())
        feature_path.append(edited.clone())
    return times, torch.stack(residual_path), torch.stack(feature_path)


@torch.inference_mode()
def flow_edit(
    source: Tensor,
    source_field: ConditionalGenerativeField,
    target_field: ConditionalGenerativeField,
    *,
    macro_steps: int,
    microsteps_per_macro: int,
    averages: int = 1,
    t_max: float = 1.0,
    t_min: float = 0.0,
    generator: torch.Generator | None = None,
    normalize_features: bool = False,
) -> tuple[Tensor, Tensor]:
    """FlowEdit's inversion-free velocity-difference Euler solver.

    The supplied fields must use the paper's rectified-flow convention:
    ``t=0`` is data and ``t=1`` is Gaussian noise. Source and target conditions
    should already be bound into the two callables. A DINO encoder alone does
    not provide either field.
    """
    if averages < 1:
        raise ValueError("averages must be positive")
    if not 0 <= t_min < t_max <= 1:
        raise ValueError("expected 0 <= t_min < t_max <= 1")
    forward = make_time_grid(
        macro_steps, microsteps_per_macro, dtype=source.dtype
    ).to(source.device)
    times = t_max - (t_max - t_min) * forward
    anchor = source
    edited = source.clone()
    path = [edited.clone()]

    for current, following in zip(times[:-1], times[1:]):
        noise = torch.randn(
            (averages, *source.shape),
            dtype=source.dtype,
            generator=generator,
        ).to(source.device)
        source_probe = (
            (1.0 - current) * anchor[None, :, :] + current * noise
        ).reshape(-1, source.shape[1])
        edited_repeated = edited.repeat(averages, 1)
        anchor_repeated = anchor.repeat(averages, 1)
        target_probe = edited_repeated + source_probe - anchor_repeated
        time_batch = current.expand(len(source_probe))
        delta = target_field(target_probe, time_batch) - source_field(
            source_probe, time_batch
        )
        delta = delta.reshape(averages, len(source), -1).mean(0)
        edited = edited + (following - current) * delta
        if normalize_features:
            edited = F.normalize(edited, dim=1)
        path.append(edited.clone())
    return times, torch.stack(path)
