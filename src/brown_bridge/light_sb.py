"""Light Schrödinger bridges in decodable low-frequency latent coordinates.

For a source law ``mu`` and target law ``nu``, LightSB parameterizes the
terminal Schrödinger potential by a positive Gaussian mixture.  With Brownian
reference kernel ``N(x, epsilon I)``, the induced endpoint conditional is

    Y | X=x ~ sum_k w_k(x) N(r_k + S_k x, epsilon S_k),

where every ``S_k`` is diagonal here.  The module is deliberately independent
of the image model: images are encoded, projected to a coarse latent coordinate,
transported, and decoded after putting the untouched source residual back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CoordinateStandardizer:
    """Invertible coordinate-wise standardization fitted to endpoint samples."""

    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, samples: Tensor, *, minimum_scale: float = 1e-4) -> "CoordinateStandardizer":
        if samples.ndim != 2 or not len(samples):
            raise ValueError("samples must have shape [samples, coordinates]")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        samples = samples.detach().float().cpu()
        return cls(
            mean=samples.mean(0),
            scale=samples.std(0, unbiased=False).clamp_min(minimum_scale),
        )

    def transform(self, samples: Tensor) -> Tensor:
        return (samples - self.mean.to(samples)) / self.scale.to(samples)

    def inverse(self, standardized: Tensor) -> Tensor:
        return standardized * self.scale.to(standardized) + self.mean.to(standardized)

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean.cpu(), "scale": self.scale.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "CoordinateStandardizer":
        if set(state) != {"mean", "scale"}:
            raise ValueError("standardizer state must contain exactly mean and scale")
        if state["mean"].shape != state["scale"].shape:
            raise ValueError("standardizer tensors must have a common shape")
        return cls(state["mean"].float().cpu(), state["scale"].float().cpu())


def coarse_latent_coordinate(latent: Tensor, *, factor: int = 4) -> Tensor:
    """Average-pool a spatial latent and flatten its transport coordinate."""
    if latent.ndim != 4 or factor < 1:
        raise ValueError("latent must be [batch, channels, height, width] and factor positive")
    if latent.shape[-2] % factor or latent.shape[-1] % factor:
        raise ValueError("latent spatial dimensions must be divisible by factor")
    return F.avg_pool2d(latent, factor, factor).flatten(1)


def lift_coarse_coordinate(
    source_latent: Tensor,
    coordinate: Tensor,
    *,
    factor: int = 4,
    residual_weight: float | Tensor = 1.0,
) -> Tensor:
    """Replace only the coarse component of ``source_latent``.

    Nearest-neighbor lifting makes average pooling a left inverse.  Thus the
    fine residual ``z - U(Pz)`` is preserved exactly and
    ``P(lift(z, c)) == c`` up to floating-point roundoff.
    """
    source_coordinate = coarse_latent_coordinate(source_latent, factor=factor)
    if coordinate.ndim != 2 or coordinate.shape != source_coordinate.shape:
        raise ValueError("coordinate shape does not match the pooled source latent")
    batch, channels, height, width = source_latent.shape
    coarse_height, coarse_width = height // factor, width // factor
    source_grid = source_coordinate.reshape(batch, channels, coarse_height, coarse_width)
    target_grid = coordinate.reshape(batch, channels, coarse_height, coarse_width)
    source_low = F.interpolate(source_grid, scale_factor=factor, mode="nearest")
    target_low = F.interpolate(target_grid, scale_factor=factor, mode="nearest")
    weight = torch.as_tensor(residual_weight, dtype=source_latent.dtype, device=source_latent.device)
    if torch.any(weight < 0) or torch.any(weight > 1):
        raise ValueError("residual_weight must lie in [0, 1]")
    return weight * (source_latent - source_low) + target_low


class DiagonalLightSB(nn.Module):
    """Diagonal Gaussian-potential Light Schrödinger bridge.

    This follows the LightSB endpoint-potential objective

        E_mu[log C_theta(X)] - E_nu[log v_theta(Y)],

    so training needs independent samples from the two marginal laws rather
    than paired examples or a precomputed coupling.
    """

    def __init__(
        self,
        dimension: int,
        *,
        components: int = 10,
        epsilon: float = 0.1,
        initial_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if dimension < 1 or components < 1 or epsilon <= 0 or initial_scale <= 0:
            raise ValueError("invalid LightSB dimensions or positive parameters")
        self.dimension = dimension
        self.components = components
        self.register_buffer("epsilon", torch.tensor(float(epsilon)))
        self.log_alpha_raw = nn.Parameter(
            torch.full((components,), float(epsilon * -math.log(components)))
        )
        self.centers = nn.Parameter(torch.randn(components, dimension))
        self.log_scale = nn.Parameter(
            torch.full((components, dimension), float(math.log(initial_scale)))
        )

    @property
    def log_alpha(self) -> Tensor:
        return self.log_alpha_raw / self.epsilon

    @property
    def scale(self) -> Tensor:
        # Bounds avoid overflow during early high-dimensional optimization.
        return self.log_scale.clamp(-9.0, 5.0).exp()

    def initialize_centers(self, target_samples: Tensor, *, generator: torch.Generator | None = None) -> None:
        if target_samples.ndim != 2 or target_samples.shape[1] != self.dimension:
            raise ValueError("target_samples has the wrong shape")
        if len(target_samples) < self.components:
            raise ValueError("not enough target samples to initialize all components")
        indices = torch.randperm(len(target_samples), generator=generator)[: self.components]
        with torch.no_grad():
            self.centers.copy_(target_samples[indices].to(self.centers))

    def conditional_logits(self, source: Tensor) -> Tensor:
        self._check_samples(source)
        quadratic = (source[:, None] * self.scale[None] * source[:, None]).sum(-1)
        linear = (source[:, None] * self.centers[None]).sum(-1)
        return (quadratic + 2.0 * linear) / (2.0 * self.epsilon) + self.log_alpha[None]

    def log_normalizer(self, source: Tensor) -> Tensor:
        return torch.logsumexp(self.conditional_logits(source), dim=1)

    def log_potential(self, target: Tensor) -> Tensor:
        self._check_samples(target)
        variance = self.epsilon * self.scale
        log_density = -0.5 * (
            self.dimension * math.log(2.0 * math.pi)
            + variance.log().sum(1)[None]
            + ((target[:, None] - self.centers[None]).square() / variance[None]).sum(2)
        )
        return torch.logsumexp(self.log_alpha[None] + log_density, dim=1)

    def objective(self, source: Tensor, target: Tensor) -> Tensor:
        return self.log_normalizer(source).mean() - self.log_potential(target).mean()

    def conditional_mean(self, source: Tensor) -> Tensor:
        weight = self.conditional_logits(source).softmax(1)
        means = self.centers[None] + self.scale[None] * source[:, None]
        return (weight[:, :, None] * means).sum(1)

    @torch.no_grad()
    def sample_endpoint(
        self,
        source: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        logits = self.conditional_logits(source)
        component = torch.multinomial(
            logits.softmax(1).cpu(), 1, generator=generator
        ).squeeze(1).to(source.device)
        selected_scale = self.scale[component]
        selected_center = self.centers[component]
        noise = torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device="cpu",
        ).to(source.device)
        return selected_center + selected_scale * source + (self.epsilon * selected_scale).sqrt() * noise

    def _check_samples(self, samples: Tensor) -> None:
        if samples.ndim != 2 or samples.shape[1] != self.dimension:
            raise ValueError(f"samples must have shape [batch, {self.dimension}]")


def brownian_bridge_path(
    source: Tensor,
    target: Tensor,
    times: Tensor,
    *,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one temporally coherent Brownian bridge for fixed endpoints.

    The reference SDE is ``dX_t = sqrt(epsilon) dW_t``.  Consequently the
    conditional covariance at time ``t`` is ``epsilon*t*(1-t) I``.
    """
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must have a common [batch, dimension] shape")
    if times.ndim != 1 or len(times) < 2 or epsilon < 0:
        raise ValueError("times must be a grid and epsilon nonnegative")
    grid = times.detach().float().cpu()
    if not torch.all(grid[1:] >= grid[:-1]) or float(grid[0]) != 0.0 or float(grid[-1]) != 1.0:
        raise ValueError("times must increase from exactly zero to one")
    dt = grid[1:] - grid[:-1]
    increments = torch.randn(
        (len(dt), *source.shape), generator=generator, dtype=source.dtype
    ) * dt.sqrt()[:, None, None]
    zero = torch.zeros(source.shape, dtype=source.dtype, device="cpu")
    motion = torch.cat([zero[None], increments.cumsum(0)], dim=0)
    bridge_noise = motion - grid[:, None, None] * motion[-1][None]
    mean = (1.0 - grid[:, None, None]) * source.cpu()[None] + grid[:, None, None] * target.cpu()[None]
    return (mean + math.sqrt(epsilon) * bridge_noise).to(source.device)


@torch.no_grad()
def expected_minibatch_ot_pairs(
    source: Tensor,
    target: Tensor,
    *,
    batch_size: int,
    batches: int,
    epsilon: float = 0.05,
    sinkhorn_iterations: int = 40,
    generator: torch.Generator | None = None,
    source_cost_features: Tensor | None = None,
    target_cost_features: Tensor | None = None,
    pairing: str = "sample",
) -> tuple[Tensor, Tensor]:
    """Sample pairs from repeated entropic minibatch-OT plans.

    Repeated random minibatches approximate the expected batch transport plan,
    rather than pretending one small batch is the population coupling.  The
    squared Euclidean cost is divided by dimension so epsilon is stable across
    transport-coordinate sizes.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be rank-two with a common dimension")
    if batch_size < 2 or batches < 1 or epsilon <= 0 or sinkhorn_iterations < 1:
        raise ValueError("invalid minibatch-OT parameters")
    if batch_size > min(len(source), len(target)):
        raise ValueError("batch_size exceeds an endpoint sample count")
    if pairing not in {"sample", "rounded"}:
        raise ValueError("pairing must be 'sample' or 'rounded'")
    if pairing == "rounded" and batches * batch_size > min(len(source), len(target)):
        raise ValueError("rounded pairing requires disjoint endpoint minibatches")
    if (source_cost_features is None) != (target_cost_features is None):
        raise ValueError("both cost-feature tensors must be supplied together")
    x, y = source.detach().float().cpu(), target.detach().float().cpu()
    cost_x, cost_y = x, y
    semantic_cost = source_cost_features is not None
    if semantic_cost:
        assert source_cost_features is not None and target_cost_features is not None
        if (
            source_cost_features.ndim != 2
            or target_cost_features.ndim != 2
            or len(source_cost_features) != len(source)
            or len(target_cost_features) != len(target)
            or source_cost_features.shape[1] != target_cost_features.shape[1]
        ):
            raise ValueError("cost features must align with endpoints and share a dimension")
        cost_x = F.normalize(source_cost_features.detach().float().cpu(), dim=1)
        cost_y = F.normalize(target_cost_features.detach().float().cpu(), dim=1)
    paired_source: list[Tensor] = []
    paired_target: list[Tensor] = []
    log_marginal = -math.log(batch_size)
    source_permutation = torch.randperm(len(x), generator=generator)
    target_permutation = torch.randperm(len(y), generator=generator)
    for batch_index in range(batches):
        if pairing == "rounded":
            start, stop = batch_index * batch_size, (batch_index + 1) * batch_size
            source_index = source_permutation[start:stop]
            target_index = target_permutation[start:stop]
        else:
            source_index = torch.randperm(len(x), generator=generator)[:batch_size]
            target_index = torch.randperm(len(y), generator=generator)[:batch_size]
        left, right = x[source_index], y[target_index]
        if semantic_cost:
            cost = 1.0 - cost_x[source_index] @ cost_y[target_index].T
        else:
            cost = torch.cdist(left, right).square() / x.shape[1]
        log_kernel = -cost / epsilon
        log_u = torch.zeros(batch_size)
        log_v = torch.zeros(batch_size)
        for _ in range(sinkhorn_iterations):
            log_u = log_marginal - torch.logsumexp(log_kernel + log_v[None], dim=1)
            log_v = log_marginal - torch.logsumexp(log_kernel.T + log_u[None], dim=1)
        plan = (log_u[:, None] + log_kernel + log_v[None]).exp()
        if pairing == "sample":
            flat = torch.multinomial(
                plan.flatten() / plan.sum(), batch_size, replacement=True, generator=generator
            )
            paired_source.append(left[torch.div(flat, batch_size, rounding_mode="floor")])
            paired_target.append(right[flat.remainder(batch_size)])
        else:
            # Greedily round the entropic plan to a permutation. Sorting by
            # transported mass is deterministic and preserves one-to-one
            # marginals without introducing a SciPy dependency.
            used_row = torch.zeros(batch_size, dtype=torch.bool)
            used_column = torch.zeros(batch_size, dtype=torch.bool)
            rows: list[int] = []
            columns: list[int] = []
            for edge in torch.argsort(plan.flatten(), descending=True).tolist():
                row, column = divmod(edge, batch_size)
                if not used_row[row] and not used_column[column]:
                    used_row[row] = True
                    used_column[column] = True
                    rows.append(row)
                    columns.append(column)
                    if len(rows) == batch_size:
                        break
            paired_source.append(left[torch.tensor(rows)])
            paired_target.append(right[torch.tensor(columns)])
    return torch.cat(paired_source), torch.cat(paired_target)
