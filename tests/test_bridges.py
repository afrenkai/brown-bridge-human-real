import torch

from brown_bridge.bridges import brownian_interpolant, sample_pinned_brownian_bridge


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


def test_brownian_interpolant_uses_per_sample_time() -> None:
    source = torch.zeros(2, 1)
    target = torch.ones(2, 1)
    noise = torch.zeros(2, 1)
    result = brownian_interpolant(
        source, target, torch.tensor([0.25, 0.75]), noise, epsilon=0.4
    )
    torch.testing.assert_close(result, torch.tensor([[0.25], [0.75]]))


def test_pinned_bridge_has_exact_endpoints_and_zero_noise_mean() -> None:
    initial = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    path = sample_pinned_brownian_bridge(
        initial,
        target,
        epsilon=0.0,
        macro_steps=4,
        microsteps_per_macro=2,
        noise=torch.randn(7, 2, 3),
    )
    expected = torch.linspace(0.0, 1.0, 5)[:, None, None].expand_as(path)
    torch.testing.assert_close(path, expected)


def test_pinned_bridge_is_stochastic_inside() -> None:
    initial = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    path = sample_pinned_brownian_bridge(
        initial,
        target,
        epsilon=0.1,
        macro_steps=4,
    )
    torch.testing.assert_close(path[0], initial)
    torch.testing.assert_close(path[-1], target)
    assert not torch.allclose(path[2], torch.full_like(path[2], 0.5))
