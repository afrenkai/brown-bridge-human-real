"""Condition image-renderer controls on latent reciprocal trajectories."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def pin_trajectory_endpoint(trajectory: Tensor, endpoint: Tensor) -> Tensor:
    """Pin a sampled path to ``endpoint`` while preserving its bridge residual.

    If ``R_t`` is the input trajectory, this returns

    ``(1-t) R_0 + t endpoint + R_t - ((1-t) R_0 + t R_1)``.

    Thus both endpoints are exact and all nonlinear/stochastic deviations from
    the raw endpoint chord remain unchanged.
    """
    if trajectory.ndim != 3:
        raise ValueError("trajectory must have shape [time, batch, features]")
    if endpoint.shape != trajectory.shape[1:]:
        raise ValueError("endpoint must have shape [batch, features]")
    time = torch.linspace(
        0.0,
        1.0,
        len(trajectory),
        device=trajectory.device,
        dtype=trajectory.dtype,
    ).view(-1, 1, 1)
    raw_chord = torch.lerp(trajectory[:1], trajectory[-1:], time)
    pinned_chord = torch.lerp(trajectory[:1], endpoint[None], time)
    return pinned_chord + trajectory - raw_chord


def endpoint_anchored_styles(
    source_style: Tensor,
    target_style: Tensor,
    correction: Tensor,
    time: Tensor,
) -> Tensor:
    """Add a learned control that vanishes at both endpoint styles."""
    if source_style.shape != target_style.shape or source_style.ndim != 2:
        raise ValueError("endpoint styles must have shape [batch, style_dim]")
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    expected = (len(source_style), len(time), source_style.shape[1])
    if correction.shape != expected:
        raise ValueError(f"correction must have shape {expected}")
    weight = time.to(source_style).view(1, -1, 1)
    chord = torch.lerp(source_style[:, None], target_style[:, None], weight)
    return chord + weight * (1.0 - weight) * correction


def endpoint_anchored_layer_styles(
    source_style: Tensor,
    target_style: Tensor,
    coefficients: Tensor,
    basis: Tensor,
    time: Tensor,
) -> Tensor:
    """Construct low-rank, endpoint-pinned styles for each decoder block.

    ``coefficients`` has shape ``[batch, time, layers, rank]`` and ``basis``
    has shape ``[batch, rank, style_dim]`` (or ``[rank, style_dim]`` shared by
    the batch). The output is ``[batch, time, layers, style_dim]``.
    """
    if source_style.shape != target_style.shape or source_style.ndim != 2:
        raise ValueError("endpoint styles must have shape [batch, style_dim]")
    if coefficients.ndim != 4 or coefficients.shape[:2] != (
        len(source_style),
        len(time),
    ):
        raise ValueError(
            "coefficients must have shape [batch, time, layers, rank]"
        )
    if basis.ndim == 2:
        basis = basis[None].expand(len(source_style), -1, -1)
    if basis.ndim != 3 or basis.shape[0] != len(source_style):
        raise ValueError("basis must have shape [batch, rank, style_dim]")
    if basis.shape[1] != coefficients.shape[3] or basis.shape[2] != source_style.shape[1]:
        raise ValueError("basis dimensions do not match coefficients/styles")
    weight = time.to(source_style).view(1, -1, 1, 1)
    chord = torch.lerp(
        source_style[:, None, None], target_style[:, None, None], weight
    ).expand(-1, -1, coefficients.shape[2], -1)
    correction = torch.einsum("btlr,brd->btld", coefficients, basis)
    return chord + weight * (1.0 - weight) * correction


def temporal_curvature_loss(styles: Tensor) -> Tensor:
    """Squared discrete acceleration of a time-indexed style control."""
    if styles.ndim < 3 or styles.shape[1] < 3:
        raise ValueError("styles must have shape [batch, time>=3, ...]")
    acceleration = styles[:, 2:] - 2.0 * styles[:, 1:-1] + styles[:, :-2]
    return acceleration.square().mean()


def cosine_bridge_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """Mean angular discrepancy between rendered and prescribed states."""
    if rendered.shape != target.shape or rendered.ndim != 2:
        raise ValueError("rendered and target must have shape [samples, features]")
    return (1.0 - F.cosine_similarity(rendered, target, dim=1)).mean()


def orthogonal_patch_loss(
    rendered_patches: Tensor,
    source_patches: Tensor,
    species_direction: Tensor,
) -> Tensor:
    """Preserve corresponding spatial tokens outside the species direction."""
    if rendered_patches.ndim != 3 or source_patches.ndim != 3:
        raise ValueError("patch tensors must have shape [batch, patches, features]")
    if source_patches.shape[0] not in (1, rendered_patches.shape[0]):
        raise ValueError("source patch batch must be one or match rendered batch")
    unit = F.normalize(species_direction.to(rendered_patches), dim=0)

    def project(value: Tensor) -> Tensor:
        projection = torch.einsum("bpd,d->bp", value, unit)[..., None] * unit
        return F.normalize(value - projection, dim=2)

    rendered = project(rendered_patches)
    source = project(source_patches).expand_as(rendered)
    return (1.0 - (rendered * source).sum(2)).mean()
