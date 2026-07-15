"""Distribution-valued paths induced by an endpoint coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

Geometry = Literal["ambient", "spherical"]


@dataclass(frozen=True)
class CoupledEndpoints:
    """Particles sampled jointly rather than independent endpoint samples."""

    source: Tensor
    target: Tensor
    source_indices: Tensor
    target_indices: Tensor

    def __post_init__(self) -> None:
        if self.source.shape != self.target.shape or self.source.ndim != 2:
            raise ValueError("source and target must share shape [particles, features]")


def make_time_grid(
    macro_steps: int,
    microsteps_per_macro: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return ``n*T + 1`` points spanning zero to one.

    Macro boundaries occur at ``k / n``. Each macro interval contains ``T``
    equal numerical integration steps, so adjacent times differ by ``1/(nT)``.
    """
    if macro_steps < 1 or microsteps_per_macro < 1:
        raise ValueError("macro_steps and microsteps_per_macro must be positive")
    return torch.linspace(
        0.0,
        1.0,
        macro_steps * microsteps_per_macro + 1,
        dtype=dtype,
    )


def sample_coupling(
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    particles: int,
    *,
    generator: torch.Generator | None = None,
) -> CoupledEndpoints:
    """Draw endpoint pairs from ``pi`` without replacing it by barycenters."""
    if particles < 1:
        raise ValueError("particles must be positive")
    if tuple(plan.shape) != (len(source), len(target)):
        raise ValueError("plan shape does not match endpoint sample counts")
    probabilities = plan.detach().to(device="cpu", dtype=torch.float64).flatten()
    if torch.any(probabilities < 0) or not torch.isfinite(probabilities).all():
        raise ValueError("plan must be finite and nonnegative")
    if float(probabilities.sum()) <= 0:
        raise ValueError("plan must have positive mass")
    flat = torch.multinomial(
        probabilities / probabilities.sum(),
        particles,
        replacement=True,
        generator=generator,
    )
    target_count = len(target)
    source_indices = torch.div(flat, target_count, rounding_mode="floor")
    target_indices = flat.remainder(target_count)
    x = source.detach().cpu()[source_indices]
    y = target.detach().cpu()[target_indices]
    return CoupledEndpoints(x, y, source_indices, target_indices)


def _orthogonal_direction(x: Tensor) -> Tensor:
    """Choose a deterministic unit tangent for nearly antipodal pairs."""
    coordinate = x.abs().argmin(dim=1, keepdim=True)
    basis = torch.zeros_like(x).scatter(1, coordinate, 1.0)
    tangent = basis - (basis * x).sum(1, keepdim=True) * x
    return F.normalize(tangent, dim=1)


def _spherical_position_velocity(
    source: Tensor,
    target: Tensor,
    times: Tensor,
) -> tuple[Tensor, Tensor]:
    x = F.normalize(source, dim=1)
    y = F.normalize(target, dim=1)
    cosine = (x * y).sum(1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    residual = y - cosine * x
    residual_norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    tangent = residual / residual_norm.clamp_min(1e-12)
    fallback = _orthogonal_direction(x)
    tangent = torch.where((residual_norm > 1e-7), tangent, fallback)

    t = times[:, None, None]
    angle = t * theta[None, :, :]
    position = torch.cos(angle) * x[None, :, :] + torch.sin(angle) * tangent[None, :, :]
    velocity = theta[None, :, :] * (
        -torch.sin(angle) * x[None, :, :] + torch.cos(angle) * tangent[None, :, :]
    )

    # When x and y coincide, the geodesic is constant and the limiting
    # velocity is zero; the arbitrary fallback tangent must not leak through.
    coincident = theta <= 1e-7
    position = torch.where(coincident[None, :, :], x[None, :, :], position)
    velocity = torch.where(coincident[None, :, :], torch.zeros_like(velocity), velocity)
    return position, velocity


def interpolate(
    endpoints: CoupledEndpoints,
    times: Tensor,
    *,
    geometry: Geometry = "spherical",
    return_velocity: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Evaluate deterministic conditional paths for sampled OT pairs.

    ``ambient`` is the Euclidean displacement interpolant. ``spherical`` is
    the constant-speed great-circle path appropriate to L2-normalized DINO
    features when the OT ground cost is squared angular distance.
    """
    t = times.detach().to(device="cpu", dtype=torch.float64).flatten()
    if len(t) < 2 or torch.any(t[1:] < t[:-1]) or float(t[0]) < 0 or float(t[-1]) > 1:
        raise ValueError("times must be sorted values in [0, 1]")
    x = endpoints.source.to(dtype=torch.float64)
    y = endpoints.target.to(dtype=torch.float64)

    if geometry == "ambient":
        position = (1.0 - t[:, None, None]) * x + t[:, None, None] * y
        velocity = (y - x).expand(len(t), -1, -1)
    elif geometry == "spherical":
        position, velocity = _spherical_position_velocity(x, y, t)
    else:
        raise ValueError(f"unsupported geometry: {geometry}")
    return (position, velocity) if return_velocity else position


def brownian_bridge(
    endpoints: CoupledEndpoints,
    times: Tensor,
    *,
    diffusivity: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample temporally correlated ambient Brownian bridges.

    The reference process is ``dX_t = sqrt(2*tau) dW_t``. Conditional on an
    endpoint pair, the path is

    ``(1-t)x0 + t*x1 + sqrt(2*tau) (W_t - t W_1)``.

    For squared-Euclidean endpoint cost, an entropic plan with ``epsilon``
    matches this reference convention when ``tau = epsilon / 4``.
    """
    if diffusivity < 0:
        raise ValueError("diffusivity cannot be negative")
    t = times.detach().to(device="cpu", dtype=torch.float64).flatten()
    if len(t) < 2 or not torch.isclose(t[0], torch.tensor(0.0, dtype=t.dtype)):
        raise ValueError("times must start at zero")
    if not torch.isclose(t[-1], torch.tensor(1.0, dtype=t.dtype)):
        raise ValueError("times must end at one")
    dt = t[1:] - t[:-1]
    if torch.any(dt <= 0):
        raise ValueError("times must be strictly increasing")

    x = endpoints.source.to(dtype=torch.float64)
    y = endpoints.target.to(dtype=torch.float64)
    increments = torch.randn(
        (len(dt), len(x), x.shape[1]),
        dtype=x.dtype,
        generator=generator,
    ) * dt.sqrt()[:, None, None]
    motion = torch.cat(
        [torch.zeros((1, *x.shape), dtype=x.dtype), increments.cumsum(0)], dim=0
    )
    pinned_motion = motion - t[:, None, None] * motion[-1]
    mean = (1.0 - t[:, None, None]) * x + t[:, None, None] * y
    return mean + (2.0 * diffusivity) ** 0.5 * pinned_motion
