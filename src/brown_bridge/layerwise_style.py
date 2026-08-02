"""Monotone, endpoint-pinned clocks for layerwise generative style paths."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MonotoneLayerwiseStyleController(nn.Module):
    """Turn one endpoint chord into a content-conditioned clock per decoder block.

    Every block remains on the same endpoint segment, but may traverse it at a
    different monotone rate.  This avoids unconstrained off-segment residuals
    and guarantees exact source and target styles at physical times zero and
    one.
    """

    def __init__(
        self,
        style_dimension: int,
        context_dimension: int,
        layers: int,
        *,
        hidden_dimension: int = 128,
        max_shift: float = 3.0,
        max_log_slope: float = math.log(4.0),
    ) -> None:
        super().__init__()
        if min(style_dimension, context_dimension, layers, hidden_dimension) < 1:
            raise ValueError("controller dimensions must be positive")
        if max_shift <= 0 or max_log_slope <= 0:
            raise ValueError("clock bounds must be positive")
        self.style_dimension = style_dimension
        self.context_dimension = context_dimension
        self.layers = layers
        self.max_shift = float(max_shift)
        self.max_log_slope = float(max_log_slope)
        self.base_shift = nn.Parameter(torch.zeros(layers))
        self.base_log_slope = nn.Parameter(torch.zeros(layers))
        self.context_net = nn.Sequential(
            nn.Linear(context_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 2 * layers),
        )
        final = self.context_net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def clock(self, context: Tensor, times: Tensor) -> Tensor:
        """Return monotone clocks with shape ``[batch,time,layers]``."""
        if context.ndim != 2 or context.shape[1] != self.context_dimension:
            raise ValueError("context has the wrong shape")
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times must be a one-dimensional grid")
        if torch.any(times < 0) or torch.any(times > 1):
            raise ValueError("times must lie in [0, 1]")
        if torch.any(times[1:] < times[:-1]):
            raise ValueError("times must be nondecreasing")
        residual_shift, residual_log_slope = self.context_net(context).chunk(2, dim=1)
        shift = self.max_shift * torch.tanh(
            self.base_shift[None] + residual_shift
        )
        log_slope = (
            self.base_log_slope[None] + residual_log_slope
        ).clamp(-self.max_log_slope, self.max_log_slope)
        slope = log_slope.exp()
        time = times.to(context).clamp(1e-6, 1.0 - 1e-6)
        logit = torch.logit(time)[None, :, None]
        value = torch.sigmoid(slope[:, None] * logit + shift[:, None])
        value = torch.where(times.to(context)[None, :, None] == 0, 0.0, value)
        value = torch.where(times.to(context)[None, :, None] == 1, 1.0, value)
        return value

    def forward(
        self,
        source_style: Tensor,
        target_style: Tensor,
        context: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return layer styles ``[batch,time,layers,style]`` and their clocks."""
        if source_style.shape != target_style.shape or source_style.ndim != 2:
            raise ValueError("endpoint styles must have a common rank-two shape")
        if source_style.shape[1] != self.style_dimension:
            raise ValueError("endpoint style dimension does not match the controller")
        if len(source_style) != len(context):
            raise ValueError("style and context batches differ")
        clock = self.clock(context, times)
        source = source_style[:, None, None]
        delta = (target_style - source_style)[:, None, None]
        return source + clock[..., None] * delta, clock


class EndpointVanishingResidualStyleController(nn.Module):
    """Add a bounded low-rank detour to a monotone layerwise style chord.

    The residual is multiplied by ``4 t (1-t)``, so both endpoints remain
    exact.  Its norm is expressed relative to the endpoint displacement,
    making ``max_residual_ratio`` comparable across source images.
    """

    def __init__(
        self,
        style_dimension: int,
        context_dimension: int,
        layers: int,
        *,
        hidden_dimension: int = 128,
        residual_rank: int = 8,
        max_residual_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        if residual_rank < 1:
            raise ValueError("residual_rank must be positive")
        if max_residual_ratio <= 0:
            raise ValueError("max_residual_ratio must be positive")
        self.style_dimension = style_dimension
        self.context_dimension = context_dimension
        self.layers = layers
        self.residual_rank = residual_rank
        self.max_residual_ratio = float(max_residual_ratio)
        self.clock_controller = MonotoneLayerwiseStyleController(
            style_dimension,
            context_dimension,
            layers,
            hidden_dimension=hidden_dimension,
        )
        self.context_embedding = nn.Sequential(
            nn.Linear(context_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
        )
        self.residual_net = nn.Sequential(
            nn.Linear(hidden_dimension + 4, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, layers * residual_rank),
        )
        final = self.residual_net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.residual_basis = nn.Parameter(
            torch.randn(layers, residual_rank, style_dimension)
            / math.sqrt(style_dimension)
        )

    def forward(
        self,
        source_style: Tensor,
        target_style: Tensor,
        context: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base, clock = self.clock_controller(
            source_style, target_style, context, times
        )
        embedding = self.context_embedding(context)[:, None].expand(
            -1, len(times), -1
        )
        time = times.to(context)[None, :, None].expand(len(context), -1, -1)
        time_features = torch.cat(
            [time, time.square(), torch.sin(math.pi * time), torch.cos(math.pi * time)],
            dim=2,
        )
        coefficients = torch.tanh(
            self.residual_net(torch.cat([embedding, time_features], dim=2))
        ).reshape(len(context), len(times), self.layers, self.residual_rank)
        basis = F.normalize(self.residual_basis, dim=2)
        residual = torch.einsum("btlr,lrd->btld", coefficients, basis)
        displacement_norm = torch.linalg.vector_norm(
            target_style - source_style, dim=1
        )
        envelope = 4.0 * time.squeeze(2) * (1.0 - time.squeeze(2))
        residual = residual * (
            self.max_residual_ratio
            * displacement_norm[:, None, None, None]
            * envelope[:, :, None, None]
            / math.sqrt(self.residual_rank)
        )
        return base + residual, clock, residual


def residual_style_regularizers(
    residual: Tensor,
    source_style: Tensor,
    target_style: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return relative residual energy and discrete temporal curvature."""
    if residual.ndim != 4:
        raise ValueError("residual must have shape [batch,time,layers,style]")
    if source_style.shape != target_style.shape or source_style.ndim != 2:
        raise ValueError("endpoint styles must have a common rank-two shape")
    if residual.shape[0] != len(source_style) or residual.shape[-1] != source_style.shape[1]:
        raise ValueError("residual and endpoint style shapes differ")
    displacement_energy = (
        (target_style - source_style).square().sum(dim=1).clamp_min(1e-8)
    )
    relative_energy = (
        residual.square().sum(dim=3) / displacement_energy[:, None, None]
    ).mean()
    if residual.shape[1] < 3:
        curvature = residual.sum() * 0.0
    else:
        second_difference = (
            residual[:, 2:] - 2.0 * residual[:, 1:-1] + residual[:, :-2]
        )
        curvature = (
            second_difference.square().sum(dim=3)
            / displacement_energy[:, None, None]
        ).mean()
    return relative_energy, curvature


def layerwise_clock_regularizer(clock: Tensor, times: Tensor) -> Tensor:
    """Penalize extreme re-timing without forcing all decoder blocks together."""
    if clock.ndim != 3 or clock.shape[1] != len(times):
        raise ValueError("clock must have shape [batch,time,layers]")
    identity = times.to(clock)[None, :, None]
    return (clock - identity).square().mean()


def clip_layerwise_clock(clock: Tensor, max_deviation: float) -> Tensor:
    """Project layer clocks into a tube around the shared-style diagonal.

    StarGAN-v2 was trained with one style shared by every decoder block.  A
    layerwise bridge leaves that training support when block clocks separate.
    This projection retains their mean progress while limiting every block's
    deviation from it.  It also preserves monotonicity and exact endpoints
    when the input clocks have those properties.
    """
    if clock.ndim != 3:
        raise ValueError("clock must have shape [batch,time,layers]")
    if max_deviation < 0:
        raise ValueError("max_deviation must be nonnegative")
    center = clock.mean(dim=2, keepdim=True)
    return torch.maximum(
        torch.minimum(clock, center + max_deviation), center - max_deviation
    ).clamp(0.0, 1.0)


def layerwise_support_regularizer(clock: Tensor, max_deviation: float) -> Tensor:
    """Soft barrier for leaving the shared-style generator support tube."""
    if clock.ndim != 3:
        raise ValueError("clock must have shape [batch,time,layers]")
    if max_deviation < 0:
        raise ValueError("max_deviation must be nonnegative")
    center = clock.mean(dim=2, keepdim=True)
    excess = F.relu((clock - center).abs() - max_deviation)
    return excess.square().mean()


def chroma_laplacian(images: Tensor) -> Tensor:
    """Return a per-pixel high-frequency chroma magnitude.

    Removing luminance before the discrete Laplacian makes the diagnostic
    especially sensitive to the isolated green/magenta pixels seen in the
    layerwise StarGAN trajectories while being comparatively insensitive to
    ordinary grayscale edges.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [batch,3,height,width]")
    if min(images.shape[-2:]) < 3:
        raise ValueError("images must be at least 3x3")
    chroma = images - images.mean(dim=1, keepdim=True)
    center = chroma[..., 1:-1, 1:-1]
    laplacian = (
        4.0 * center
        - chroma[..., :-2, 1:-1]
        - chroma[..., 2:, 1:-1]
        - chroma[..., 1:-1, :-2]
        - chroma[..., 1:-1, 2:]
    )
    # The floor keeps the derivative finite at achromatic, locally constant
    # pixels; sqrt'(0) otherwise turns a perfectly valid zero loss into NaNs.
    return (laplacian.square().mean(dim=1) + 1e-12).sqrt()


def localized_chroma_artifact_loss(
    interior: Tensor,
    endpoints: Tensor,
    *,
    topk_fraction: float = 0.01,
    margin: float = 1.1,
) -> Tensor:
    """Penalize the worst localized chroma spikes above endpoint statistics.

    ``interior`` is ``[batch,time,3,height,width]`` and ``endpoints`` is
    ``[batch,2,3,height,width]``.  Only the worst ``topk_fraction`` of pixels
    contribute, avoiding the dilution suffered by an image-wide mean TV loss.
    """
    if interior.ndim != 5 or endpoints.ndim != 5:
        raise ValueError("interior and endpoints must be rank five")
    if interior.shape[0] != endpoints.shape[0] or endpoints.shape[1] != 2:
        raise ValueError("endpoints must have shape [batch,2,3,height,width]")
    if interior.shape[2:] != endpoints.shape[2:]:
        raise ValueError("interior and endpoint image shapes differ")
    if not 0 < topk_fraction <= 1:
        raise ValueError("topk_fraction must lie in (0, 1]")
    if margin < 0:
        raise ValueError("margin must be nonnegative")

    batch, time_count = interior.shape[:2]
    interior_energy = chroma_laplacian(interior.flatten(0, 1)).reshape(
        batch, time_count, -1
    )
    endpoint_energy = chroma_laplacian(endpoints.flatten(0, 1)).reshape(
        batch, 2, -1
    )
    pixel_count = interior_energy.shape[-1]
    count = max(1, int(round(topk_fraction * pixel_count)))
    endpoint_tail = endpoint_energy.topk(count, dim=2).values
    threshold = endpoint_tail[..., -1].amax(dim=1)
    excess = F.relu(interior_energy - margin * threshold[:, None, None])
    return excess.topk(count, dim=2).values.mean()
