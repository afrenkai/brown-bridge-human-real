"""Feature extraction independent of a particular dataset or notebook."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    preprocess: Callable[..., Tensor],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect normalized feature rows and the corresponding dataset indices."""
    model.eval().to(device)
    feature_batches: list[Tensor] = []
    index_batches: list[Tensor] = []
    for images, indices in loader:
        images = preprocess(images.to(device, non_blocking=True), training=False)
        features = model.forward_features(images)
        if features.ndim != 2:
            raise ValueError("forward_features must return [batch, feature_dim]")
        feature_batches.append(F.normalize(features.float(), dim=1).cpu())
        index_batches.append(indices.cpu())
    return torch.cat(feature_batches), torch.cat(index_batches)


@torch.inference_mode()
def collect_spatial_features(
    model: nn.Module,
    loader: DataLoader,
    preprocess: Callable[..., Tensor],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Collect normalized global features and spatial DINO patch tokens."""
    if not hasattr(model, "forward_spatial_features"):
        raise TypeError("model must expose forward_spatial_features")
    model.eval().to(device)
    global_batches: list[Tensor] = []
    patch_batches: list[Tensor] = []
    index_batches: list[Tensor] = []
    for images, indices in loader:
        images = preprocess(images.to(device, non_blocking=True), training=False)
        global_features, patch_features = model.forward_spatial_features(images)
        if global_features.ndim != 2 or patch_features.ndim != 3:
            raise ValueError("expected [batch, dim] globals and [batch, patches, dim]")
        global_batches.append(F.normalize(global_features.float(), dim=1).cpu())
        patch_batches.append(F.normalize(patch_features.float(), dim=2).cpu())
        index_batches.append(indices.cpu())
    return (
        torch.cat(global_batches),
        torch.cat(patch_batches),
        torch.cat(index_batches),
    )


def species_direction(source: Tensor, target: Tensor) -> Tensor:
    """Return the unit difference of empirical feature means."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be [samples, common_dimension]")
    direction = target.float().mean(0) - source.float().mean(0)
    if float(torch.linalg.vector_norm(direction)) <= 1e-8:
        raise ValueError("empirical class means do not define a species direction")
    return F.normalize(direction, dim=0)


def project_off_direction(features: Tensor, direction: Tensor) -> Tensor:
    """Project the last feature axis onto the species-orthogonal subspace."""
    if features.shape[-1] != direction.numel():
        raise ValueError("feature and direction dimensions differ")
    unit = F.normalize(direction.to(features), dim=0)
    return features - torch.einsum("...d,d->...", features, unit)[..., None] * unit


def content_descriptor(
    global_features: Tensor,
    patch_features: Tensor,
    direction: Tensor,
    *,
    global_weight: float = 1.0,
    patch_weight: float = 1.0,
) -> Tensor:
    """Build a spatial descriptor with the species mean direction removed."""
    if global_weight < 0 or patch_weight < 0 or global_weight + patch_weight <= 0:
        raise ValueError("descriptor weights must be nonnegative with positive sum")
    if global_features.ndim != 2 or patch_features.ndim != 3:
        raise ValueError("expected [samples, dim] globals and [samples, patches, dim]")
    if len(global_features) != len(patch_features):
        raise ValueError("global and patch sample counts differ")
    global_content = F.normalize(
        project_off_direction(global_features.float(), direction), dim=1
    )
    patch_content = F.normalize(
        project_off_direction(patch_features.float(), direction), dim=2
    )
    patch_count = patch_content.shape[1]
    pieces: list[Tensor] = []
    if global_weight:
        pieces.append(global_weight**0.5 * global_content)
    if patch_weight:
        pieces.append(
            (patch_weight / patch_count) ** 0.5 * patch_content.flatten(1)
        )
    return torch.cat(pieces, dim=1)


def compact_spatial_content_descriptor(
    global_features: Tensor,
    patch_features: Tensor,
    direction: Tensor,
) -> Tensor:
    """Species-orthogonal global plus 2x2-pooled spatial DINO content.

    The compact representation retains coarse layout while making online
    mini-batch OT practical for a full AFHQ training set.
    """
    if global_features.ndim != 2 or patch_features.ndim != 3:
        raise ValueError("expected [samples, dim] globals and [samples, patches, dim]")
    if len(global_features) != len(patch_features):
        raise ValueError("global and patch sample counts differ")
    patch_count = patch_features.shape[1]
    grid = round(patch_count**0.5)
    if grid * grid != patch_count or grid % 2:
        raise ValueError("patch count must form an even square grid")
    global_content = F.normalize(
        project_off_direction(global_features.float(), direction), dim=1
    )
    patch_content = project_off_direction(patch_features.float(), direction)
    spatial = patch_content.transpose(1, 2).reshape(len(patch_content), -1, grid, grid)
    spatial = F.avg_pool2d(spatial, kernel_size=2, stride=2)
    spatial = spatial.flatten(2).transpose(1, 2)
    spatial = F.normalize(spatial, dim=2)
    return torch.cat(
        [global_content, spatial.flatten(1) / spatial.shape[1] ** 0.5], dim=1
    )


def content_cost_matrix(source_descriptor: Tensor, target_descriptor: Tensor) -> Tensor:
    """Squared Euclidean content cost with stable nonnegative arithmetic."""
    if source_descriptor.ndim != 2 or target_descriptor.ndim != 2:
        raise ValueError("descriptors must be rank two")
    if source_descriptor.shape[1] != target_descriptor.shape[1]:
        raise ValueError("descriptor dimensions differ")
    source = source_descriptor.to(device="cpu", dtype=torch.float64)
    target = target_descriptor.to(device="cpu", dtype=torch.float64)
    cost = (
        source.square().sum(1, keepdim=True)
        + target.square().sum(1)[None, :]
        - 2.0 * source @ target.T
    )
    return cost.clamp_min_(0)
