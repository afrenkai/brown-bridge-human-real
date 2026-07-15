"""Spatial lifting of a global reciprocal path onto DINO patch tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .ot import barycentric_projection, sinkhorn_from_cost


def square_patch_positions(
    patch_count: int, *, dtype: torch.dtype = torch.float32
) -> Tensor:
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError("patch_count must form a square grid")
    axis = torch.linspace(-1.0, 1.0, side, dtype=dtype)
    vertical, horizontal = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([horizontal.flatten(), vertical.flatten()], dim=1)


def positional_patch_transport(
    source: Tensor,
    target: Tensor,
    *,
    position_weight: float = 0.25,
    epsilon: float = 0.08,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map target tokens barycentrically onto source-grid locations."""
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must have shape [patches, features]")
    if position_weight < 0:
        raise ValueError("position_weight must be nonnegative")
    source_unit = F.normalize(source.float(), dim=1)
    target_unit = F.normalize(target.float(), dim=1)
    feature_cost = (1.0 - source_unit @ target_unit.T).clamp_min(0.0)
    positions = square_patch_positions(len(source), dtype=feature_cost.dtype)
    position_cost = torch.cdist(positions, positions).square()
    cost = feature_cost.cpu() + position_weight * position_cost
    result = sinkhorn_from_cost(
        cost,
        epsilon=epsilon,
        max_iterations=2_000,
        tolerance=1e-9,
    )
    mapped = barycentric_projection(
        result.plan,
        source_unit,
        target_unit,
        direction="forward",
        normalize_output=True,
    ).float()
    return mapped, result.plan.float(), cost.float()


def foreground_patch_mask(
    source: Tensor,
    mapped_target: Tensor,
    species_direction: Tensor,
) -> Tensor:
    """Soft animal foreground estimate combining location and feature change."""
    if source.ndim != 2 or source.shape != mapped_target.shape:
        raise ValueError("patch endpoints must have shape [patches, features]")
    positions = square_patch_positions(len(source), dtype=source.dtype)
    horizontal, vertical = positions[:, 0], positions[:, 1]
    central = torch.exp(-1.5 * (horizontal.square() + 0.7 * vertical.square()))
    source_unit = F.normalize(source.float(), dim=1)
    target_unit = F.normalize(mapped_target.float(), dim=1)
    change = 1.0 - (source_unit * target_unit).sum(1)
    unit_species = F.normalize(species_direction.float(), dim=0)
    species_change = ((target_unit - source_unit) @ unit_species).abs()

    def rescale(value: Tensor) -> Tensor:
        return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)

    mask = 0.55 * rescale(central) + 0.25 * rescale(change) + 0.20 * rescale(species_change)
    return mask.clamp(0.05, 0.95)


def species_clock(global_path: Tensor, species_direction: Tensor) -> Tensor:
    """Monotone normalized class coordinate of one global bridge path."""
    if global_path.ndim != 2:
        raise ValueError("global_path must have shape [time, features]")
    unit = F.normalize(species_direction.to(global_path), dim=0)
    coordinate = global_path @ unit
    denominator = coordinate[-1] - coordinate[0]
    if abs(float(denominator)) < 1e-7:
        return torch.linspace(0.0, 1.0, len(global_path), device=global_path.device)
    progress = ((coordinate - coordinate[0]) / denominator).clamp(0.0, 1.0)
    progress = torch.cummax(progress, dim=0).values
    progress[0] = 0.0
    progress[-1] = 1.0
    return progress


def lift_patch_trajectory(
    source: Tensor, mapped_target: Tensor, progress: Tensor
) -> Tensor:
    """Lift scalar SDE progress to a normalized patch-token trajectory."""
    if source.ndim != 2 or source.shape != mapped_target.shape:
        raise ValueError("patch endpoints must have shape [patches, features]")
    if progress.ndim != 1:
        raise ValueError("progress must be one-dimensional")
    weight = progress.to(source).view(-1, 1, 1)
    trajectory = torch.lerp(source[None], mapped_target[None], weight)
    return F.normalize(trajectory, dim=2)


def masked_spatial_bridge_loss(
    rendered: Tensor,
    target_path: Tensor,
    source: Tensor,
    foreground_mask: Tensor,
) -> Tensor:
    """Transport foreground patches while preserving source background tokens."""
    if rendered.ndim != 3 or rendered.shape != target_path.shape:
        raise ValueError("rendered/target must have shape [time, patches, features]")
    if source.shape != rendered.shape[1:]:
        raise ValueError("source must have shape [patches, features]")
    if foreground_mask.shape != (rendered.shape[1],):
        raise ValueError("foreground_mask must have one value per patch")
    rendered_unit = F.normalize(rendered, dim=2)
    target_unit = F.normalize(target_path.to(rendered), dim=2)
    source_unit = F.normalize(source.to(rendered), dim=1)[None]
    target_error = 1.0 - (rendered_unit * target_unit).sum(2)
    source_error = 1.0 - (rendered_unit * source_unit).sum(2)
    mask = foreground_mask.to(rendered).view(1, -1)
    return (mask * target_error + (1.0 - mask) * source_error).mean()
