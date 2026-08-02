"""Spatial current/score fields and reciprocal latent SDE integrators."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .flow_matching import SpatialTimeConditionedField


class SpatialVelocityScoreField(nn.Module):
    """U-ViT with joint current-velocity and scaled-score heads.

    The second head predicts ``gamma(t) * grad log rho_t``.  This remains
    finite at the bridge endpoints, unlike the unscaled score.
    """

    def __init__(
        self,
        latent_channels: int,
        spatial_size: int,
        *,
        base_channels: int = 64,
        time_dim: int = 128,
        transformer_blocks: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.feature_dim = latent_channels * spatial_size * spatial_size
        self.backbone = SpatialTimeConditionedField(
            latent_channels,
            spatial_size,
            base_channels=base_channels,
            time_dim=time_dim,
            transformer_blocks=transformer_blocks,
            transformer_heads=transformer_heads,
            output_channels=2 * latent_channels,
        )

    def forward(self, state: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        joint = self.backbone(state, time).reshape(
            len(state), 2 * self.latent_channels, self.spatial_size, self.spatial_size
        )
        velocity, scaled_score = joint.chunk(2, dim=1)
        return velocity.flatten(1), scaled_score.flatten(1)


class AnchoredSpatialVelocityScoreField(nn.Module):
    """Source-conditioned bidirectional U-ViT current/score field.

    State, source anchor, and a constant direction plane are concatenated at
    the input.  The shared network therefore retains instance information even
    when many unpaired OT paths cross in the middle of the bridge.
    """

    def __init__(
        self,
        latent_channels: int,
        spatial_size: int,
        *,
        base_channels: int = 64,
        time_dim: int = 128,
        transformer_blocks: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.feature_dim = latent_channels * spatial_size * spatial_size
        self.input_channels = 2 * latent_channels + 1
        self.backbone = SpatialTimeConditionedField(
            self.input_channels,
            spatial_size,
            base_channels=base_channels,
            time_dim=time_dim,
            transformer_blocks=transformer_blocks,
            transformer_heads=transformer_heads,
            output_channels=2 * latent_channels,
        )

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        anchor: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if state.shape != anchor.shape or state.ndim != 2:
            raise ValueError("state and anchor must share a flattened batched shape")
        if state.shape[1] != self.feature_dim:
            raise ValueError(f"state feature dimension must be {self.feature_dim}")
        if time.numel() != len(state) or direction.numel() != len(state):
            raise ValueError("time and direction must have one value per state")
        spatial_shape = (-1, self.latent_channels, self.spatial_size, self.spatial_size)
        state_spatial = state.reshape(spatial_shape)
        anchor_spatial = anchor.reshape(spatial_shape)
        direction_plane = direction.to(state).reshape(-1, 1, 1, 1).expand(
            -1, 1, self.spatial_size, self.spatial_size
        )
        conditioned = torch.cat(
            [state_spatial, anchor_spatial, direction_plane], dim=1
        ).flatten(1)
        physical_time = torch.where(direction.to(time) >= 0, time, 1.0 - time)
        joint = self.backbone(conditioned, physical_time).reshape(
            len(state), 2 * self.latent_channels, self.spatial_size, self.spatial_size
        )
        velocity, scaled_score = joint.chunk(2, dim=1)
        travel_velocity = direction.to(velocity).reshape(-1, 1, 1, 1) * velocity
        return travel_velocity.flatten(1), scaled_score.flatten(1)


def bridge_score_scale(
    time: Tensor,
    *,
    interpolant_epsilon: float,
    minimum_time: float = 1e-3,
) -> Tensor:
    """Return ``sqrt(epsilon t(1-t))`` with a numerical boundary cutoff."""
    if interpolant_epsilon <= 0 or not 0 < minimum_time < 0.5:
        raise ValueError("invalid interpolant epsilon or minimum time")
    physical_time = time.clamp(minimum_time, 1.0 - minimum_time)
    return torch.sqrt(interpolant_epsilon * physical_time * (1.0 - physical_time))


def bridge_diffusion_rate(time: Tensor, *, peak_rate: float) -> Tensor:
    """Endpoint-vanishing SDE rate with maximum ``peak_rate`` at time 1/2."""
    if peak_rate < 0:
        raise ValueError("peak_rate must be nonnegative")
    return peak_rate * 4.0 * time.clamp(0.0, 1.0) * (1.0 - time.clamp(0.0, 1.0))


@torch.inference_mode()
def integrate_probability_flow(
    field: SpatialVelocityScoreField,
    initial: Tensor,
    *,
    macro_steps: int,
    microsteps_per_macro: int = 1,
    reverse: bool = False,
) -> Tensor:
    """Heun integration of the current velocity, used as the mean-path check."""
    if macro_steps < 1 or microsteps_per_macro < 1:
        raise ValueError("step counts must be positive")
    total_steps = macro_steps * microsteps_per_macro
    times = torch.linspace(0.0, 1.0, total_steps + 1, device=initial.device)
    if reverse:
        times = 1.0 - times
    state = initial
    trajectory = [state.clone()]
    for index, (left, right) in enumerate(zip(times[:-1], times[1:]), start=1):
        dt = right - left
        left_time = left.expand(len(state))
        right_time = right.expand(len(state))
        velocity, _ = field(state, left_time)
        proposal = state + dt * velocity
        corrected, _ = field(proposal, right_time)
        state = state + 0.5 * dt * (velocity + corrected)
        if index % microsteps_per_macro == 0:
            trajectory.append(state.clone())
    return torch.stack(trajectory)


@torch.inference_mode()
def integrate_reciprocal_sde(
    field: SpatialVelocityScoreField,
    initial: Tensor,
    *,
    interpolant_epsilon: float,
    peak_diffusion_rate: float,
    macro_steps: int,
    microsteps_per_macro: int = 1,
    reverse: bool = False,
    noise: Tensor | None = None,
) -> Tensor:
    """Euler--Maruyama integration of the forward or backward Nelson drift.

    For a time-only rate ``kappa(t)``, both

    ``b+ = v + kappa*s/2`` and ``b- = v - kappa*s/2``

    have the stochastic-interpolant marginals.  The reverse process uses the
    backward drift with negative physical ``dt``.
    """
    if macro_steps < 1 or microsteps_per_macro < 1:
        raise ValueError("step counts must be positive")
    total_steps = macro_steps * microsteps_per_macro
    if noise is None:
        noise = torch.randn((total_steps, *initial.shape), device=initial.device)
    if tuple(noise.shape) != (total_steps, *initial.shape):
        raise ValueError("noise has the wrong shape")
    times = torch.linspace(0.0, 1.0, total_steps + 1, device=initial.device)
    if reverse:
        times = 1.0 - times
    state = initial
    trajectory = [state.clone()]
    for index, (left, right) in enumerate(zip(times[:-1], times[1:]), start=1):
        dt = right - left
        time = left.expand(len(state))
        velocity, scaled_score = field(state, time)
        gamma = bridge_score_scale(
            time,
            interpolant_epsilon=interpolant_epsilon,
        )[:, None]
        score = scaled_score / gamma
        rate = bridge_diffusion_rate(time, peak_rate=peak_diffusion_rate)[:, None]
        drift = velocity + (-0.5 if reverse else 0.5) * rate * score
        state = state + dt * drift + torch.sqrt(rate * dt.abs()) * noise[index - 1]
        if index % microsteps_per_macro == 0:
            trajectory.append(state.clone())
    return torch.stack(trajectory)


@torch.inference_mode()
def integrate_anchored_probability_flow(
    field: AnchoredSpatialVelocityScoreField,
    initial: Tensor,
    *,
    direction: int,
    macro_steps: int,
    microsteps_per_macro: int = 1,
) -> Tensor:
    """Heun mean path for a source-anchored direction-conditioned bridge."""
    if direction not in (-1, 1) or macro_steps < 1 or microsteps_per_macro < 1:
        raise ValueError("invalid direction or step counts")
    total_steps = macro_steps * microsteps_per_macro
    times = torch.linspace(0.0, 1.0, total_steps + 1, device=initial.device)
    anchor = initial.clone()
    sign = torch.full((len(initial),), float(direction), device=initial.device)
    state = initial
    trajectory = [state.clone()]
    for index, (left, right) in enumerate(zip(times[:-1], times[1:]), start=1):
        dt = right - left
        velocity, _ = field(state, left.expand(len(state)), anchor, sign)
        proposal = state + dt * velocity
        corrected, _ = field(proposal, right.expand(len(state)), anchor, sign)
        state = state + 0.5 * dt * (velocity + corrected)
        if index % microsteps_per_macro == 0:
            trajectory.append(state.clone())
    return torch.stack(trajectory)


@torch.inference_mode()
def integrate_anchored_sde(
    field: AnchoredSpatialVelocityScoreField,
    initial: Tensor,
    *,
    direction: int,
    interpolant_epsilon: float,
    peak_diffusion_rate: float,
    macro_steps: int,
    microsteps_per_macro: int = 1,
    noise: Tensor | None = None,
) -> Tensor:
    """Euler--Maruyama path for a source-conditioned forward travel clock."""
    if direction not in (-1, 1) or macro_steps < 1 or microsteps_per_macro < 1:
        raise ValueError("invalid direction or step counts")
    total_steps = macro_steps * microsteps_per_macro
    if noise is None:
        noise = torch.randn((total_steps, *initial.shape), device=initial.device)
    if tuple(noise.shape) != (total_steps, *initial.shape):
        raise ValueError("noise has the wrong shape")
    times = torch.linspace(0.0, 1.0, total_steps + 1, device=initial.device)
    anchor = initial.clone()
    sign = torch.full((len(initial),), float(direction), device=initial.device)
    state = initial
    trajectory = [state.clone()]
    for index, (left, right) in enumerate(zip(times[:-1], times[1:]), start=1):
        dt = right - left
        time = left.expand(len(state))
        velocity, scaled_score = field(state, time, anchor, sign)
        gamma = bridge_score_scale(
            time, interpolant_epsilon=interpolant_epsilon
        )[:, None]
        score = scaled_score / gamma
        rate = bridge_diffusion_rate(time, peak_rate=peak_diffusion_rate)[:, None]
        drift = velocity + 0.5 * rate * score
        state = state + dt * drift + torch.sqrt(rate * dt) * noise[index - 1]
        if index % microsteps_per_macro == 0:
            trajectory.append(state.clone())
    return torch.stack(trajectory)


@torch.inference_mode()
def sample_pinned_brownian_bridge(
    initial: Tensor,
    target: Tensor,
    *,
    epsilon: float,
    macro_steps: int,
    microsteps_per_macro: int = 1,
    noise: Tensor | None = None,
) -> Tensor:
    """Sample exact grid transitions of a Brownian bridge pinned at ``target``.

    This is the Doob h-transform

    ``dX_t = (target - X_t)/(1-t) dt + sqrt(epsilon) dW_t``.

    The finite-step conditional transition is Gaussian and the final endpoint
    is imposed exactly, avoiding Euler instability in the singular drift.
    """
    if initial.shape != target.shape or initial.ndim != 2:
        raise ValueError("initial and target must share a flattened batched shape")
    if epsilon < 0 or macro_steps < 1 or microsteps_per_macro < 1:
        raise ValueError("invalid epsilon or step counts")
    total_steps = macro_steps * microsteps_per_macro
    if noise is None:
        noise = torch.randn((total_steps - 1, *initial.shape), device=initial.device)
    if tuple(noise.shape) != (total_steps - 1, *initial.shape):
        raise ValueError("noise has the wrong shape")
    state = initial
    trajectory = [state.clone()]
    for index in range(total_steps):
        left = index / total_steps
        right = (index + 1) / total_steps
        if index == total_steps - 1:
            state = target
        else:
            fraction = (right - left) / (1.0 - left)
            mean = state + fraction * (target - state)
            variance = epsilon * (right - left) * (1.0 - right) / (1.0 - left)
            state = mean + variance**0.5 * noise[index]
        if (index + 1) % microsteps_per_macro == 0:
            trajectory.append(state.clone())
    return torch.stack(trajectory)
