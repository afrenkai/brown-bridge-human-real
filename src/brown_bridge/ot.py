"""Stable entropic optimal transport for empirical feature measures.

The solver uses the KL-to-product-measure convention

    min_pi <pi, C> + epsilon KL(pi || a tensor b),

which has the same optimizer as the more common negative-entropy convention
when the marginals are fixed, while making the probabilistic interpretation
of a Schrödinger endpoint coupling explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

Cost = Literal["cosine", "sqeuclidean", "angular_squared"]


@dataclass(frozen=True)
class SinkhornResult:
    plan: Tensor
    cost_matrix: Tensor
    transport_cost: Tensor
    regularized_objective: Tensor
    kl_to_product: Tensor
    row_marginal_error: Tensor
    column_marginal_error: Tensor
    iterations: int
    converged: bool


def _probability_weights(
    count: int,
    weights: Tensor | None,
    *,
    dtype: torch.dtype,
) -> Tensor:
    if weights is None:
        return torch.full((count,), 1.0 / count, dtype=dtype)
    weights = weights.detach().to(device="cpu", dtype=dtype).flatten()
    if len(weights) != count:
        raise ValueError(f"expected {count} weights, received {len(weights)}")
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("weights must be finite and strictly positive")
    return weights / weights.sum()


def pairwise_cost(source: Tensor, target: Tensor, cost: Cost) -> Tensor:
    """Return the ground-cost matrix on CPU in float64."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be rank-two [samples, features]")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if not len(source) or not len(target):
        raise ValueError("source and target cannot be empty")

    x = source.detach().to(device="cpu", dtype=torch.float64)
    y = target.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("features must be finite")

    if cost == "sqeuclidean":
        return torch.cdist(x, y).square().clamp_min_(0)

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    cosine = (x @ y.T).clamp(-1.0, 1.0)
    if cost == "cosine":
        # On the unit sphere this is one half of squared chordal distance.
        return (1.0 - cosine).clamp_min_(0)
    if cost == "angular_squared":
        # Squared geodesic distance on the unit hypersphere.
        return torch.acos(cosine).square()
    raise ValueError(f"unsupported transport cost: {cost}")


def sinkhorn(
    source: Tensor,
    target: Tensor,
    *,
    epsilon: float = 0.05,
    max_iterations: int = 1_000,
    tolerance: float = 1e-7,
    cost: Cost = "angular_squared",
    source_weights: Tensor | None = None,
    target_weights: Tensor | None = None,
) -> SinkhornResult:
    """Solve balanced entropic OT with log-domain Sinkhorn iterations."""
    matrix = pairwise_cost(source, target, cost)
    return sinkhorn_from_cost(
        matrix,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
        source_weights=source_weights,
        target_weights=target_weights,
    )


def sinkhorn_from_cost(
    cost_matrix: Tensor,
    *,
    epsilon: float = 0.05,
    max_iterations: int = 1_000,
    tolerance: float = 1e-7,
    source_weights: Tensor | None = None,
    target_weights: Tensor | None = None,
) -> SinkhornResult:
    """Solve balanced entropic OT from a precomputed ground-cost matrix."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    if cost_matrix.ndim != 2 or not all(cost_matrix.shape):
        raise ValueError("cost_matrix must be nonempty and rank two")
    matrix = cost_matrix.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(matrix).all() or torch.any(matrix < 0):
        raise ValueError("cost_matrix must be finite and nonnegative")
    a = _probability_weights(matrix.shape[0], source_weights, dtype=matrix.dtype)
    b = _probability_weights(matrix.shape[1], target_weights, dtype=matrix.dtype)
    log_a, log_b = a.log(), b.log()
    log_kernel = -matrix / epsilon
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    converged = False
    row_error = torch.tensor(float("inf"), dtype=matrix.dtype)
    column_error = torch.tensor(float("inf"), dtype=matrix.dtype)

    for iteration in range(1, max_iterations + 1):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel.T + log_u[None, :], dim=1)

        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            log_plan = log_u[:, None] + log_kernel + log_v[None, :]
            plan = log_plan.exp()
            row_error = (plan.sum(1) - a).abs().max()
            column_error = (plan.sum(0) - b).abs().max()
            if max(float(row_error), float(column_error)) <= tolerance:
                converged = True
                break

    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = log_plan.exp()
    log_product = log_a[:, None] + log_b[None, :]
    kl = torch.sum(torch.where(plan > 0, plan * (log_plan - log_product), 0.0))
    transport_cost = torch.sum(plan * matrix)

    return SinkhornResult(
        plan=plan,
        cost_matrix=matrix,
        transport_cost=transport_cost,
        regularized_objective=transport_cost + epsilon * kl,
        kl_to_product=kl,
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        iterations=iteration,
        converged=converged,
    )


def sinkhorn_divergence(
    source: Tensor,
    target: Tensor,
    *,
    epsilon: float = 0.05,
    max_iterations: int = 1_000,
    tolerance: float = 1e-7,
    cost: Cost = "angular_squared",
    source_weights: Tensor | None = None,
    target_weights: Tensor | None = None,
) -> tuple[Tensor, SinkhornResult]:
    """Return the debiased Sinkhorn divergence and the cross coupling."""
    common = {
        "epsilon": epsilon,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "cost": cost,
    }
    cross = sinkhorn(
        source,
        target,
        source_weights=source_weights,
        target_weights=target_weights,
        **common,
    )
    source_self = sinkhorn(
        source,
        source,
        source_weights=source_weights,
        target_weights=source_weights,
        **common,
    )
    target_self = sinkhorn(
        target,
        target,
        source_weights=target_weights,
        target_weights=target_weights,
        **common,
    )
    divergence = (
        cross.regularized_objective
        - 0.5 * source_self.regularized_objective
        - 0.5 * target_self.regularized_objective
    )
    return divergence, cross


def barycentric_projection(
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    *,
    direction: Literal["forward", "backward"] = "forward",
    normalize_output: bool = False,
) -> Tensor:
    """Return a conditional-mean summary of a coupling.

    A barycentric projection is generally not measure preserving. Use sampled
    coupling pairs from :func:`brown_bridge.paths.sample_coupling` when the
    endpoint distribution must remain the target empirical measure.
    """
    pi = plan.detach().to(device="cpu", dtype=torch.float64)
    x = source.detach().to(device="cpu", dtype=torch.float64)
    y = target.detach().to(device="cpu", dtype=torch.float64)
    if tuple(pi.shape) != (len(x), len(y)):
        raise ValueError("plan shape does not match endpoint sample counts")

    if direction == "forward":
        result = pi @ y / pi.sum(1, keepdim=True).clamp_min(1e-15)
    elif direction == "backward":
        result = pi.T @ x / pi.sum(0, keepdim=True).T.clamp_min(1e-15)
    else:
        raise ValueError(f"unsupported direction: {direction}")
    return F.normalize(result, dim=1) if normalize_output else result
