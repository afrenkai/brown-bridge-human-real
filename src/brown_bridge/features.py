"""Content features used by the RAE endpoint-coupled diagnostic."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def species_direction(source: Tensor, target: Tensor) -> Tensor:
    """Return the unit difference of two empirical feature means."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be [samples, common_dimension]")
    direction = target.float().mean(0) - source.float().mean(0)
    if float(torch.linalg.vector_norm(direction)) <= 1e-8:
        raise ValueError("empirical class means do not define a direction")
    return functional.normalize(direction, dim=0)


def project_off_direction(features: Tensor, direction: Tensor) -> Tensor:
    """Project the final feature axis away from a unit direction."""
    if features.shape[-1] != direction.numel():
        raise ValueError("feature and direction dimensions differ")
    unit_direction = functional.normalize(direction.to(features), dim=0)
    coefficients = torch.einsum("...d,d->...", features, unit_direction)
    return features - coefficients[..., None] * unit_direction


def compact_spatial_content_descriptor(
    global_features: Tensor,
    patch_features: Tensor,
    direction: Tensor,
) -> Tensor:
    """Combine projected global features with pooled spatial features."""
    if global_features.ndim != 2 or patch_features.ndim != 3:
        raise ValueError("expected [samples, dim] globals and [samples, patches, dim]")
    if len(global_features) != len(patch_features):
        raise ValueError("global and patch sample counts differ")
    patch_count = patch_features.shape[1]
    grid_size = round(patch_count**0.5)
    if grid_size * grid_size != patch_count or grid_size % 2:
        raise ValueError("patch count must form an even square grid")

    global_content = functional.normalize(
        project_off_direction(global_features.float(), direction), dim=1
    )
    patch_content = project_off_direction(patch_features.float(), direction)
    spatial_content = patch_content.transpose(1, 2).reshape(
        len(patch_content), -1, grid_size, grid_size
    )
    spatial_content = functional.avg_pool2d(spatial_content, kernel_size=2, stride=2)
    spatial_content = spatial_content.flatten(2).transpose(1, 2)
    spatial_content = functional.normalize(spatial_content, dim=2)
    scaled_spatial_content = (
        spatial_content.flatten(1) / spatial_content.shape[1] ** 0.5
    )
    return torch.cat([global_content, scaled_spatial_content], dim=1)


def content_cost_matrix(
    source_descriptor: Tensor,
    target_descriptor: Tensor,
) -> Tensor:
    """Return a stable squared-Euclidean descriptor cost matrix."""
    if source_descriptor.ndim != 2 or target_descriptor.ndim != 2:
        raise ValueError("descriptors must be rank two")
    if source_descriptor.shape[1] != target_descriptor.shape[1]:
        raise ValueError("descriptor dimensions differ")
    source_values = source_descriptor.to(device="cpu", dtype=torch.float64)
    target_values = target_descriptor.to(device="cpu", dtype=torch.float64)
    costs = (
        source_values.square().sum(1, keepdim=True)
        + target_values.square().sum(1)[None, :]
        - 2.0 * source_values @ target_values.T
    )
    return costs.clamp_min_(0)
