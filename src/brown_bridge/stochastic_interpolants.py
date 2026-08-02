"""Simulation-free stochastic interpolants between two empirical measures.

The default interpolant is the Brownian bridge mixture

    X_t = (1-t) X_0 + t X_1 + sqrt(epsilon t(1-t)) Z,

where ``(X_0, X_1)`` is sampled from an endpoint coupling and ``Z`` is standard
Gaussian.  Regressing the pathwise time derivative and conditional Gaussian
score yields the marginal current velocity and score.  Those two fields define
both a probability-flow ODE and Nelson's forward/backward diffusions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .paths import sample_coupling


@dataclass(frozen=True)
class StochasticInterpolantBatch:
    """One simulation-free regression batch."""

    time: Tensor
    state: Tensor
    velocity_target: Tensor
    score_target: Tensor
    source: Tensor
    target: Tensor
    noise: Tensor


def _column_time(time: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(time, dtype=reference.dtype, device=reference.device)
    if value.ndim == 0:
        value = value.expand(len(reference))
    if value.ndim != 1 or len(value) != len(reference):
        raise ValueError("time must be scalar or have one entry per sample")
    return value[:, None]


def brownian_interpolant(
    source: Tensor,
    target: Tensor,
    time: Tensor | float,
    noise: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Evaluate a Brownian bridge mixture with fixed endpoints and noise."""
    if source.shape != target.shape or source.shape != noise.shape or source.ndim < 2:
        raise ValueError("source, target, and noise must share a batched shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    t = _column_time(time, source)
    shape = (len(source),) + (1,) * (source.ndim - 1)
    t = t.reshape(shape)
    if torch.any(t < 0) or torch.any(t > 1):
        raise ValueError("time must lie in [0, 1]")
    gamma = torch.sqrt(torch.as_tensor(epsilon, device=source.device, dtype=source.dtype) * t * (1.0 - t))
    return (1.0 - t) * source + t * target + gamma * noise


def brownian_regression_targets(
    source: Tensor,
    target: Tensor,
    time: Tensor | float,
    noise: Tensor,
    *,
    epsilon: float,
    minimum_time: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Return pathwise velocity and conditional-score regression targets.

    The score diverges at the exact endpoints, so training times must remain in
    ``[minimum_time, 1-minimum_time]``.  Endpoint samples themselves remain
    exact because the noise amplitude vanishes at zero and one.
    """
    if source.shape != target.shape or source.shape != noise.shape or source.ndim < 2:
        raise ValueError("source, target, and noise must share a batched shape")
    if epsilon <= 0 or not 0 < minimum_time < 0.5:
        raise ValueError("epsilon and minimum_time must be positive")
    t = _column_time(time, source)
    shape = (len(source),) + (1,) * (source.ndim - 1)
    t = t.reshape(shape)
    if torch.any(t < minimum_time) or torch.any(t > 1.0 - minimum_time):
        raise ValueError("training times are too close to an endpoint")
    variance = torch.as_tensor(epsilon, device=source.device, dtype=source.dtype) * t * (1.0 - t)
    gamma = variance.sqrt()
    velocity = target - source + epsilon * (1.0 - 2.0 * t) * noise / (2.0 * gamma)
    score = -noise / gamma
    return velocity, score


def make_stochastic_interpolant_batch(
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    batch_size: int,
    *,
    epsilon: float,
    minimum_time: float = 1e-3,
    endpoint_bias: float = 0.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> StochasticInterpolantBatch:
    """Sample coupled endpoints and exact simulation-free regression targets."""
    if not 0.0 <= endpoint_bias <= 1.0:
        raise ValueError("endpoint_bias must lie in [0, 1]")
    endpoints = sample_coupling(plan, source, target, batch_size, generator=generator)
    x0, x1 = endpoints.source.float(), endpoints.target.float()
    uniform_time = torch.rand(batch_size, generator=generator)
    if endpoint_bias:
        # sin^2(pi U / 2) has the arcsine law and therefore allocates more
        # samples to the endpoint boundary layers without introducing atoms.
        endpoint_time = torch.sin(0.5 * torch.pi * torch.rand(
            batch_size, generator=generator
        )).square()
        choose_endpoint = torch.rand(batch_size, generator=generator) < endpoint_bias
        unit_time = torch.where(choose_endpoint, endpoint_time, uniform_time)
    else:
        unit_time = uniform_time
    time = minimum_time + (1.0 - 2.0 * minimum_time) * unit_time
    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
    state = brownian_interpolant(x0, x1, time, noise, epsilon=epsilon)
    velocity, score = brownian_regression_targets(
        x0,
        x1,
        time,
        noise,
        epsilon=epsilon,
        minimum_time=minimum_time,
    )
    if device is not None:
        time = time.to(device)
        state, velocity, score = state.to(device), velocity.to(device), score.to(device)
        x0, x1, noise = x0.to(device), x1.to(device), noise.to(device)
    return StochasticInterpolantBatch(time, state, velocity, score, x0, x1, noise)


def score_flow_matching_loss(
    velocity_prediction: Tensor,
    score_prediction: Tensor,
    batch: StochasticInterpolantBatch,
    *,
    score_weight: float = 1.0,
) -> Tensor:
    """Joint flow- and score-matching objective used by stochastic bridges."""
    if score_weight < 0:
        raise ValueError("score_weight must be nonnegative")
    return F.mse_loss(velocity_prediction, batch.velocity_target) + score_weight * F.mse_loss(
        score_prediction, batch.score_target
    )


def nelson_drifts(
    velocity: Tensor,
    score: Tensor,
    *,
    diffusion_rate: float,
) -> tuple[Tensor, Tensor]:
    """Return forward and backward drifts for noise ``sqrt(rate) dW``.

    With current velocity ``v`` and density score ``s = grad log rho_t``,

        b+ = v + rate*s/2,    b- = v - rate*s/2,

    hence ``b+ - b- = rate*s`` (Nelson's relation).  Both directions share the
    same one-time marginals as the probability-flow ODE ``dX = v dt``.
    """
    if velocity.shape != score.shape:
        raise ValueError("velocity and score must have the same shape")
    if diffusion_rate < 0:
        raise ValueError("diffusion_rate must be nonnegative")
    osmotic = 0.5 * diffusion_rate * score
    return velocity + osmotic, velocity - osmotic


def time_reversal_loss(
    forward_velocity: Tensor,
    reverse_velocity: Tensor,
    forward_score: Tensor,
    reverse_score: Tensor,
) -> Tensor:
    """Penalize violation of reciprocal time-reversal symmetry.

    Reverse predictions are evaluated at the same physical state and at
    reversed clock time.  Current velocity changes sign; the marginal score
    does not.
    """
    tensors = (forward_velocity, reverse_velocity, forward_score, reverse_score)
    if any(value.shape != forward_velocity.shape for value in tensors[1:]):
        raise ValueError("all predictions must share a shape")
    return F.mse_loss(forward_velocity, -reverse_velocity) + F.mse_loss(
        forward_score, reverse_score
    )


def sliced_wasserstein_squared(
    source: Tensor,
    target: Tensor,
    *,
    directions: Tensor,
) -> Tensor:
    """Differentiable sliced-Wasserstein terminal-law discrepancy.

    Every normalized direction defines a one-dimensional optimal transport
    problem, which is solved exactly by sorting equally weighted projections.
    Fixed directions avoid injecting Monte Carlo noise into short rollout
    fine-tuning.  Equal sample counts are intentional: use matched minibatches.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be rank-two with a common dimension")
    if not len(source) or len(source) != len(target):
        raise ValueError("source and target must have the same nonzero sample count")
    if directions.ndim != 2 or directions.shape[1] != source.shape[1] or not len(directions):
        raise ValueError("directions must be [projections, dimension]")
    directions = directions.to(source)
    norms = torch.linalg.vector_norm(directions, dim=1)
    if torch.any(norms <= 0):
        raise ValueError("projection directions must be nonzero")
    unit = directions / norms[:, None]
    source_quantiles = torch.sort(source @ unit.T, dim=0).values
    target_quantiles = torch.sort(target @ unit.T, dim=0).values
    return (source_quantiles - target_quantiles).square().mean()
