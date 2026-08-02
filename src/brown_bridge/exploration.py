"""Training-time candidate exploration objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor


@dataclass(frozen=True)
class BestOfKResult:
    """Best candidate losses and the selected candidate for each example."""

    loss: Tensor
    candidate_losses: Tensor
    winner_indices: Tensor


def best_of_k_loss(
    candidate_losses: Tensor,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> BestOfKResult:
    """Keep the lowest loss among K candidates for every batch example."""
    if candidate_losses.ndim != 2:
        raise ValueError("candidate_losses must have shape [batch, candidates]")
    if candidate_losses.shape[1] < 1:
        raise ValueError("candidate_losses must contain at least one candidate")
    if not candidate_losses.is_floating_point():
        raise ValueError("candidate_losses must use a floating-point dtype")
    if not candidate_losses.isfinite().all():
        raise ValueError("candidate_losses must be finite")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unsupported reduction: {reduction}")

    winner_indices = candidate_losses.argmin(dim=1)
    best_losses = candidate_losses.gather(1, winner_indices[:, None]).squeeze(1)
    if reduction == "mean":
        loss = best_losses.mean()
    elif reduction == "sum":
        loss = best_losses.sum()
    else:
        loss = best_losses
    return BestOfKResult(
        loss=loss,
        candidate_losses=candidate_losses,
        winner_indices=winner_indices,
    )
