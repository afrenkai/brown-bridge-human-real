import torch

from brown_bridge.stochastic_interpolants import (
    brownian_interpolant,
    brownian_regression_targets,
    make_stochastic_interpolant_batch,
    sliced_wasserstein_squared,
    nelson_drifts,
    time_reversal_loss,
)


def test_brownian_interpolant_has_exact_endpoints() -> None:
    source = torch.randn(5, 3)
    target = torch.randn(5, 3)
    noise = torch.randn(5, 3)
    torch.testing.assert_close(
        brownian_interpolant(source, target, 0.0, noise, epsilon=0.4), source
    )
    torch.testing.assert_close(
        brownian_interpolant(source, target, 1.0, noise, epsilon=0.4), target
    )


def test_velocity_target_is_path_derivative() -> None:
    source = torch.randn(7, 4)
    target = torch.randn(7, 4)
    noise = torch.randn(7, 4)
    time = torch.linspace(0.1, 0.9, 7)
    velocity, _ = brownian_regression_targets(
        source, target, time, noise, epsilon=0.3
    )
    step = 1e-4
    right = brownian_interpolant(source, target, time + step, noise, epsilon=0.3)
    left = brownian_interpolant(source, target, time - step, noise, epsilon=0.3)
    torch.testing.assert_close((right - left) / (2 * step), velocity, atol=2e-3, rtol=2e-3)


def test_simulation_free_batch_and_nelson_relation() -> None:
    source = torch.randn(12, 2)
    target = torch.randn(15, 2)
    plan = torch.ones(12, 15) / (12 * 15)
    batch = make_stochastic_interpolant_batch(
        plan,
        source,
        target,
        32,
        epsilon=0.2,
        generator=torch.Generator().manual_seed(2),
    )
    assert batch.state.shape == (32, 2)
    forward, backward = nelson_drifts(
        batch.velocity_target, batch.score_target, diffusion_rate=0.2
    )
    torch.testing.assert_close(forward - backward, 0.2 * batch.score_target)


def test_endpoint_biased_times_and_sliced_wasserstein_gradient() -> None:
    source = torch.randn(20, 2)
    target = source.detach().clone()
    plan = torch.eye(20) / 20
    batch = make_stochastic_interpolant_batch(
        plan,
        source,
        target,
        2_000,
        epsilon=0.2,
        minimum_time=0.01,
        endpoint_bias=1.0,
        generator=torch.Generator().manual_seed(9),
    )
    assert ((batch.time < 0.1) | (batch.time > 0.9)).float().mean() > 0.35

    generated = source.detach().clone().requires_grad_()
    directions = torch.randn(32, 2)
    loss = sliced_wasserstein_squared(generated, target + 0.2, directions=directions)
    loss.backward()
    assert loss > 0
    assert generated.grad is not None and generated.grad.abs().sum() > 0
    torch.testing.assert_close(
        sliced_wasserstein_squared(source, source, directions=directions),
        source.new_zeros(()),
        atol=1e-7,
        rtol=0,
    )


def test_time_reversal_loss_vanishes_for_reciprocal_fields() -> None:
    velocity = torch.randn(8, 3)
    score = torch.randn(8, 3)
    assert time_reversal_loss(velocity, -velocity, score, score).item() == 0.0
