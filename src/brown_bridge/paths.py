"""Sampling from empirical endpoint couplings."""

from __future__ import annotations

import torch
from torch import Tensor

from .schemas import CoupledEndpoints


def sample_coupling(
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    particles: int,
    *,
    generator: torch.Generator | None = None,
) -> CoupledEndpoints:
    """Draw endpoint pairs from a transport plan without using barycenters."""
    if particles < 1:
        raise ValueError("particles must be positive")
    if tuple(plan.shape) != (len(source), len(target)):
        raise ValueError("plan shape does not match endpoint sample counts")
    probabilities = plan.detach().to(device="cpu", dtype=torch.float64).flatten()
    if torch.any(probabilities < 0) or not torch.isfinite(probabilities).all():
        raise ValueError("plan must be finite and nonnegative")
    if float(probabilities.sum()) <= 0:
        raise ValueError("plan must have positive mass")

    flat_indices = torch.multinomial(
        probabilities / probabilities.sum(),
        particles,
        replacement=True,
        generator=generator,
    )
    target_count = len(target)
    source_indices = torch.div(flat_indices, target_count, rounding_mode="floor")
    target_indices = flat_indices.remainder(target_count)
    source_samples = source.detach().cpu()[source_indices]
    target_samples = target.detach().cpu()[target_indices]
    return CoupledEndpoints(
        source_samples,
        target_samples,
        source_indices,
        target_indices,
    )
