from __future__ import annotations

import torch
from torch import Tensor


def brownian_interpolant(
    source: Tensor,
    target: Tensor,
    time: Tensor | float,
    noise: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Evaluate a Brownian-bridge mixture with fixed endpoints and noise."""
    if source.shape != target.shape or source.shape != noise.shape or source.ndim < 2:
        raise ValueError("source, target, and noise must share a batched shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    time_values = torch.as_tensor(time, dtype=source.dtype, device=source.device)
    if time_values.ndim == 0:
        time_values = time_values.expand(len(source))
    if time_values.ndim != 1 or len(time_values) != len(source):
        raise ValueError("time must be scalar or have one entry per sample")
    broadcast_shape = (len(source),) + (1,) * (source.ndim - 1)
    time_values = time_values.reshape(broadcast_shape)
    if torch.any(time_values < 0) or torch.any(time_values > 1):
        raise ValueError("time must lie in [0, 1]")

    variance = epsilon * time_values * (1.0 - time_values)
    return (1.0 - time_values) * source + time_values * target + variance.sqrt() * noise


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
    """Sample exact grid transitions of a bridge pinned at ``target``."""
    if initial.shape != target.shape or initial.ndim != 2:
        raise ValueError("initial and target must share a flattened batched shape")
    if epsilon < 0 or macro_steps < 1 or microsteps_per_macro < 1:
        raise ValueError("invalid epsilon or step counts")

    total_steps = macro_steps * microsteps_per_macro
    expected_noise_shape = (total_steps - 1, *initial.shape)
    innovations = noise
    if innovations is None:
        innovations = torch.randn(expected_noise_shape, device=initial.device)
    if tuple(innovations.shape) != expected_noise_shape:
        raise ValueError("noise has the wrong shape")

    state = initial
    trajectory = [state.clone()]
    for step_index in range(total_steps):
        left_time = step_index / total_steps
        right_time = (step_index + 1) / total_steps
        if step_index == total_steps - 1:
            state = target
        else:
            fraction = (right_time - left_time) / (1.0 - left_time)
            mean = state + fraction * (target - state)
            variance = (
                epsilon
                * (right_time - left_time)
                * (1.0 - right_time)
                / (1.0 - left_time)
            )
            state = mean + variance**0.5 * innovations[step_index]
        if (step_index + 1) % microsteps_per_macro == 0:
            trajectory.append(state.clone())
    return torch.stack(trajectory)
