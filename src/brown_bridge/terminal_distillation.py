"""Utilities for distilling an empirical OT terminal kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.inference_mode()
def topk_terminal_support(
    source_descriptor: Tensor,
    target_descriptor: Tensor,
    source_layout: Tensor,
    target_layout: Tensor,
    *,
    neighbors: int,
    layout_weight: float,
    chunk_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Return the lowest-cost opposite-domain support points per source.

    Cost is a convex combination of cosine distances in species-projected
    spatial-DINO content and standardized coarse KL layout.
    """
    tensors = (source_descriptor, target_descriptor, source_layout, target_layout)
    if any(value.ndim != 2 or not len(value) for value in tensors):
        raise ValueError("descriptors and layouts must be nonempty matrices")
    if source_descriptor.shape[1] != target_descriptor.shape[1]:
        raise ValueError("descriptor dimensions differ")
    if source_layout.shape[1] != target_layout.shape[1]:
        raise ValueError("layout dimensions differ")
    if len(source_descriptor) != len(source_layout):
        raise ValueError("source descriptor/layout counts differ")
    if len(target_descriptor) != len(target_layout):
        raise ValueError("target descriptor/layout counts differ")
    if not 1 <= neighbors <= len(target_descriptor):
        raise ValueError("neighbors exceeds target support")
    if not 0.0 <= layout_weight <= 1.0 or chunk_size < 1:
        raise ValueError("invalid layout weight or chunk size")
    source_descriptor = F.normalize(source_descriptor.float().cpu(), dim=1)
    target_descriptor = F.normalize(target_descriptor.float().cpu(), dim=1)
    source_layout = F.normalize(source_layout.float().cpu(), dim=1)
    target_layout = F.normalize(target_layout.float().cpu(), dim=1)
    index_batches: list[Tensor] = []
    cost_batches: list[Tensor] = []
    for start in range(0, len(source_descriptor), chunk_size):
        descriptor = source_descriptor[start : start + chunk_size]
        layout = source_layout[start : start + chunk_size]
        cost = (
            (1.0 - layout_weight) * (1.0 - descriptor @ target_descriptor.T)
            + layout_weight * (1.0 - layout @ target_layout.T)
        )
        values, indices = torch.topk(cost, neighbors, largest=False, sorted=True)
        index_batches.append(indices)
        cost_batches.append(values)
    return torch.cat(index_batches), torch.cat(cost_batches)


def sample_terminal_indices(
    support_indices: Tensor,
    support_costs: Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one cached terminal per row from a Gibbs top-k kernel."""
    if support_indices.shape != support_costs.shape or support_indices.ndim != 2:
        raise ValueError("support indices and costs must share a matrix shape")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if temperature == 0:
        choice = torch.zeros(len(support_indices), dtype=torch.long)
    else:
        probabilities = torch.softmax(-support_costs.float() / temperature, dim=1)
        choice = torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).squeeze(1)
    return support_indices[torch.arange(len(support_indices)), choice]
