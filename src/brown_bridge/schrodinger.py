"""High-level empirical Schrödinger bridge on DINO feature measures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .ot import SinkhornResult, sinkhorn
from .paths import CoupledEndpoints, brownian_bridge, make_time_grid, sample_coupling


@dataclass(frozen=True)
class BridgeSample:
    times: Tensor
    paths: Tensor
    endpoints: CoupledEndpoints


@dataclass(frozen=True)
class EmpiricalSchrodingerBridge:
    """A reciprocal mixture with prescribed empirical endpoint marginals.

    The reference diffusion is ``dX_t = sqrt(2*tau) dW_t`` on a unit horizon.
    With squared-Euclidean ground cost, ``epsilon = 4*tau``. The entropy-
    regularized endpoint plan is mixed with Brownian bridges conditional on
    every sampled endpoint pair.
    """

    source: Tensor
    target: Tensor
    coupling: SinkhornResult
    epsilon: float
    diffusivity: float

    @classmethod
    def fit(
        cls,
        source: Tensor,
        target: Tensor,
        *,
        epsilon: float,
        source_weights: Tensor | None = None,
        target_weights: Tensor | None = None,
        max_iterations: int = 1_000,
        tolerance: float = 1e-7,
    ) -> "EmpiricalSchrodingerBridge":
        coupling = sinkhorn(
            source,
            target,
            epsilon=epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
            cost="sqeuclidean",
            source_weights=source_weights,
            target_weights=target_weights,
        )
        return cls(
            source=source.detach().cpu(),
            target=target.detach().cpu(),
            coupling=coupling,
            epsilon=epsilon,
            diffusivity=epsilon / 4.0,
        )

    def sample_paths(
        self,
        particles: int,
        *,
        macro_steps: int,
        microsteps_per_macro: int,
        generator: torch.Generator | None = None,
    ) -> BridgeSample:
        endpoints = sample_coupling(
            self.coupling.plan,
            self.source,
            self.target,
            particles,
            generator=generator,
        )
        times = make_time_grid(macro_steps, microsteps_per_macro)
        paths = brownian_bridge(
            endpoints,
            times,
            diffusivity=self.diffusivity,
            generator=generator,
        )
        return BridgeSample(times, paths, endpoints)

    def conditional_marginal_mean(self, time: float) -> Tensor:
        """Return means of pairwise bridge marginals for every coupling atom."""
        if not 0 <= time <= 1:
            raise ValueError("time must lie in [0, 1]")
        source_index, target_index = torch.meshgrid(
            torch.arange(len(self.source)),
            torch.arange(len(self.target)),
            indexing="ij",
        )
        x = self.source[source_index.flatten()].to(torch.float64)
        y = self.target[target_index.flatten()].to(torch.float64)
        return (1.0 - time) * x + time * y

    def conditional_marginal_variance(self, time: float) -> float:
        """Per-coordinate variance of each reciprocal Brownian component."""
        if not 0 <= time <= 1:
            raise ValueError("time must lie in [0, 1]")
        return 2.0 * self.diffusivity * time * (1.0 - time)
