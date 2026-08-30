"""Balanced entropic optimal transport for empirical measures."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from .schemas import SinkhornResult, TransportCost


def normalize_probability_weights(
    count: int,
    weights: Tensor | None,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Validate and normalize strictly positive empirical weights."""
    if count < 1:
        raise ValueError("count must be positive")
    if weights is None:
        return torch.full((count,), 1.0 / count, dtype=dtype)
    normalized = weights.detach().to(device="cpu", dtype=dtype).flatten()
    if len(normalized) != count:
        raise ValueError(f"expected {count} weights, received {len(normalized)}")
    if not torch.isfinite(normalized).all() or torch.any(normalized <= 0):
        raise ValueError("weights must be finite and strictly positive")
    return normalized / normalized.sum()


def pairwise_cost(
    source: Tensor,
    target: Tensor,
    cost: TransportCost,
) -> Tensor:
    """Return a float64 CPU ground-cost matrix."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be rank-two [samples, features]")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if len(source) == 0 or len(target) == 0:
        raise ValueError("source and target cannot be empty")

    source_values = source.detach().to(device="cpu", dtype=torch.float64)
    target_values = target.detach().to(device="cpu", dtype=torch.float64)
    if (
        not torch.isfinite(source_values).all()
        or not torch.isfinite(target_values).all()
    ):
        raise ValueError("features must be finite")
    if cost == "sqeuclidean":
        return torch.cdist(source_values, target_values).square().clamp_min_(0)

    source_values = functional.normalize(source_values, dim=1)
    target_values = functional.normalize(target_values, dim=1)
    cosine = (source_values @ target_values.T).clamp(-1.0, 1.0)
    if cost == "cosine":
        return (1.0 - cosine).clamp_min_(0)
    if cost == "angular_squared":
        return torch.acos(cosine).square()
    raise ValueError(f"unsupported transport cost: {cost}")


def sinkhorn(
    source: Tensor,
    target: Tensor,
    *,
    epsilon: float = 0.05,
    max_iterations: int = 1_000,
    tolerance: float = 1e-7,
    cost: TransportCost = "angular_squared",
    source_weights: Tensor | None = None,
    target_weights: Tensor | None = None,
) -> SinkhornResult:
    """Solve balanced entropic OT with log-domain Sinkhorn iterations."""
    return sinkhorn_from_cost(
        pairwise_cost(source, target, cost),
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
    """Solve balanced entropic OT from a ground-cost matrix."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    if cost_matrix.ndim != 2 or min(cost_matrix.shape) < 1:
        raise ValueError("cost_matrix must be nonempty and rank two")

    matrix = cost_matrix.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(matrix).all() or torch.any(matrix < 0):
        raise ValueError("cost_matrix must be finite and nonnegative")
    source_mass = normalize_probability_weights(
        matrix.shape[0], source_weights, dtype=matrix.dtype
    )
    target_mass = normalize_probability_weights(
        matrix.shape[1], target_weights, dtype=matrix.dtype
    )
    log_source_mass = source_mass.log()
    log_target_mass = target_mass.log()
    log_kernel = -matrix / epsilon
    log_source_scaling = torch.zeros_like(log_source_mass)
    log_target_scaling = torch.zeros_like(log_target_mass)
    converged = False
    row_error = torch.tensor(float("inf"), dtype=matrix.dtype)
    column_error = torch.tensor(float("inf"), dtype=matrix.dtype)

    for iteration in range(1, max_iterations + 1):
        log_source_scaling = log_source_mass - torch.logsumexp(
            log_kernel + log_target_scaling[None, :], dim=1
        )
        log_target_scaling = log_target_mass - torch.logsumexp(
            log_kernel.T + log_source_scaling[None, :], dim=1
        )
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            log_plan = (
                log_source_scaling[:, None] + log_kernel + log_target_scaling[None, :]
            )
            plan = log_plan.exp()
            row_error = (plan.sum(1) - source_mass).abs().max()
            column_error = (plan.sum(0) - target_mass).abs().max()
            if max(float(row_error), float(column_error)) <= tolerance:
                converged = True
                break

    log_plan = log_source_scaling[:, None] + log_kernel + log_target_scaling[None, :]
    plan = log_plan.exp()
    log_product_mass = log_source_mass[:, None] + log_target_mass[None, :]
    kl_to_product = torch.sum(
        torch.where(plan > 0, plan * (log_plan - log_product_mass), 0.0)
    )
    transport_cost = torch.sum(plan * matrix)
    return SinkhornResult(
        plan=plan,
        cost_matrix=matrix,
        transport_cost=transport_cost,
        regularized_objective=transport_cost + epsilon * kl_to_product,
        kl_to_product=kl_to_product,
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        iterations=iteration,
        converged=converged,
    )
