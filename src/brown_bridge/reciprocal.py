"""Bidirectional stochastic-interpolant fields with reciprocal regularization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .paths import sample_coupling


@dataclass(frozen=True)
class ReciprocalBatch:
    state: Tensor
    time: Tensor
    direction: Tensor
    conditional_velocity: Tensor
    scaled_score: Tensor
    flow_weight: Tensor


@dataclass
class InequalityAugmentedLagrangian:
    """Powell--Hestenes--Rockafellar penalty for ``value <= threshold``.

    The multiplier is dual state, not a differentiable model parameter. Call
    :meth:`primal_loss` while building the model loss and :meth:`dual_step`
    after the corresponding optimizer step.
    """

    threshold: float
    penalty: float
    multiplier: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold < 0 or self.penalty <= 0 or self.multiplier < 0:
            raise ValueError("threshold/multiplier must be nonnegative and penalty positive")

    def primal_loss(self, value: Tensor) -> Tensor:
        violation = value - self.threshold
        shifted = F.relu(value.new_tensor(self.multiplier) + self.penalty * violation)
        return (shifted.square() - self.multiplier**2) / (2.0 * self.penalty)

    def dual_step(self, value: Tensor | float) -> float:
        scalar = float(value.detach()) if isinstance(value, Tensor) else float(value)
        self.multiplier = max(
            0.0, self.multiplier + self.penalty * (scalar - self.threshold)
        )
        return self.multiplier


class DirectionalTimeMLP(nn.Module):
    """One field shared by both directions, conditioned by a sign in {-1,+1}."""

    def __init__(
        self,
        feature_dim: int,
        *,
        width: int = 512,
        depth: int = 4,
        time_frequencies: int = 4,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(time_frequencies, dtype=torch.float32),
            persistent=False,
        )
        input_dim = feature_dim + 2 + 2 * time_frequencies
        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor, time: Tensor, direction: Tensor) -> Tensor:
        time = time.reshape(-1, 1).to(state)
        direction = direction.reshape(-1, 1).to(state)
        phase = torch.pi * time * self.frequencies.to(state)
        embedding = torch.cat([time, direction, phase.sin(), phase.cos()], dim=1)
        return self.network(torch.cat([state, embedding], dim=1))


def make_reciprocal_batch(
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    batch_size: int,
    *,
    diffusivity: float,
    generator: torch.Generator,
    device: torch.device,
) -> ReciprocalBatch:
    """Sample both time directions from one reciprocal Brownian mixture."""
    if diffusivity <= 0:
        raise ValueError("diffusivity must be positive")
    endpoints = sample_coupling(plan, source, target, batch_size, generator=generator)
    direction = 2 * torch.randint(0, 2, (batch_size,), generator=generator) - 1
    forward = direction > 0
    left = torch.where(forward[:, None], endpoints.source, endpoints.target)
    right = torch.where(forward[:, None], endpoints.target, endpoints.source)

    uniform = torch.rand(batch_size, generator=generator)
    near_zero = torch.rand(batch_size, generator=generator).pow(4.0)
    near_one = 1.0 - torch.rand(batch_size, generator=generator).pow(4.0)
    selector = torch.randint(0, 3, (batch_size,), generator=generator)
    time = torch.where(
        selector == 0, uniform, torch.where(selector == 1, near_zero, near_one)
    )
    time = 0.01 + 0.98 * time
    noise = torch.randn(batch_size, source.shape[1], generator=generator)
    gamma = torch.sqrt(2.0 * diffusivity * time * (1.0 - time)).clamp_min(1e-6)
    gamma_dot = diffusivity * (1.0 - 2.0 * time) / gamma
    mean = (1.0 - time[:, None]) * left + time[:, None] * right
    state = mean + gamma[:, None] * noise
    velocity = right - left + gamma_dot[:, None] * noise
    flow_weight = gamma.square() / (gamma.square() + 0.0025)
    return ReciprocalBatch(
        state.to(device),
        time.to(device),
        direction.to(device),
        velocity.to(device),
        (-noise).to(device),
        flow_weight.to(device),
    )


def orthogonal_displacement(state: Tensor, reference: Tensor, species: Tensor) -> Tensor:
    """Mean squared displacement orthogonal to the unit species direction."""
    unit = F.normalize(species.to(state), dim=0)
    delta = state - reference
    residual = delta - (delta @ unit)[:, None] * unit
    return residual.square().mean()


def species_progression_loss(
    velocity: Tensor,
    direction: Tensor,
    species: Tensor,
    *,
    margin: float = 0.0,
) -> Tensor:
    """Penalize only wrong-way species motion, without rewarding overshoot."""
    unit = F.normalize(species.to(velocity), dim=0)
    signed_derivative = direction.to(velocity) * (velocity @ unit)
    return F.relu(margin - signed_derivative).square().mean()


def integrate_directional_sde(
    velocity: DirectionalTimeMLP,
    score: DirectionalTimeMLP,
    initial: Tensor,
    *,
    direction: int,
    steps: int,
    kappa: float,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Differentiable stochastic Heun integration on a unit time interval."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    if steps < 1 or kappa < 0:
        raise ValueError("steps must be positive and kappa nonnegative")
    state = initial
    dt = 1.0 / steps
    if noise is None:
        noise = torch.randn(
            (steps, *initial.shape),
            generator=generator,
            device="cpu",
            dtype=initial.dtype,
        ).to(initial.device)
    elif tuple(noise.shape) != (steps, *initial.shape):
        raise ValueError("noise must have shape [steps, batch, features]")
    sign = torch.full((len(state),), float(direction), device=state.device)
    diffusion_scale = (2.0 * kappa * dt) ** 0.5
    trajectory = [state]
    for index in range(steps):
        left = torch.full((len(state),), index * dt, device=state.device)
        right = left + dt
        drift = velocity(state, left, sign)
        if kappa:
            drift = drift + kappa * score(state, left, sign)
        increment = diffusion_scale * noise[index]
        proposal = state + dt * drift + increment
        corrected = velocity(proposal, right, sign)
        if kappa:
            corrected = corrected + kappa * score(proposal, right, sign)
        state = state + 0.5 * dt * (drift + corrected) + increment
        trajectory.append(state)
    return torch.stack(trajectory)


def shared_noise_cycle_loss(
    velocity: DirectionalTimeMLP,
    score: DirectionalTimeMLP,
    source: Tensor,
    *,
    steps: int,
    kappa: float,
    generator: torch.Generator,
    direction: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Integrate one direction and its reciprocal with reversed increments."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    noise = torch.randn(
        (steps, *source.shape), generator=generator, dtype=source.dtype
    ).to(source.device)
    forward = integrate_directional_sde(
        velocity,
        score,
        source,
        direction=direction,
        steps=steps,
        kappa=kappa,
        noise=noise,
    )
    reverse = integrate_directional_sde(
        velocity,
        score,
        forward[-1],
        direction=-direction,
        steps=steps,
        kappa=kappa,
        noise=-noise.flip(0),
    )
    return F.mse_loss(reverse[-1], source), forward, reverse
