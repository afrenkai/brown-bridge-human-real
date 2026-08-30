"""Typed results shared by transport domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor

TransportCost = Literal["cosine", "sqeuclidean", "angular_squared"]
Reduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class SinkhornResult:
    """A balanced entropic optimal-transport solution."""

    plan: Tensor
    cost_matrix: Tensor
    transport_cost: Tensor
    regularized_objective: Tensor
    kl_to_product: Tensor
    row_marginal_error: Tensor
    column_marginal_error: Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class CoupledEndpoints:
    """Particles sampled jointly from an endpoint coupling."""

    source: Tensor
    target: Tensor
    source_indices: Tensor
    target_indices: Tensor

    def __post_init__(self) -> None:
        if self.source.shape != self.target.shape or self.source.ndim != 2:
            raise ValueError("source and target must share shape [particles, features]")


@dataclass(frozen=True)
class BestOfKResult:
    """Best candidate losses and selected candidate indices."""

    loss: Tensor
    candidate_losses: Tensor
    winner_indices: Tensor
