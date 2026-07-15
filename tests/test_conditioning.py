import torch

from brown_bridge.conditioning import (
    endpoint_anchored_layer_styles,
    endpoint_anchored_styles,
    pin_trajectory_endpoint,
    temporal_curvature_loss,
)


def test_pinning_preserves_endpoints_and_residual() -> None:
    trajectory = torch.randn(7, 3, 5)
    endpoint = torch.randn(3, 5)
    pinned = pin_trajectory_endpoint(trajectory, endpoint)
    assert torch.allclose(pinned[0], trajectory[0], atol=1e-6)
    assert torch.allclose(pinned[-1], endpoint, atol=1e-6)
    time = torch.linspace(0, 1, 7).view(-1, 1, 1)
    raw_residual = trajectory - torch.lerp(trajectory[:1], trajectory[-1:], time)
    pinned_residual = pinned - torch.lerp(pinned[:1], pinned[-1:], time)
    assert torch.allclose(pinned_residual, raw_residual, atol=1e-6)


def test_endpoint_style_correction_vanishes() -> None:
    source = torch.randn(2, 8)
    target = torch.randn(2, 8)
    time = torch.linspace(0, 1, 5)
    correction = torch.randn(2, 5, 8)
    styles = endpoint_anchored_styles(source, target, correction, time)
    assert torch.allclose(styles[:, 0], source)
    assert torch.allclose(styles[:, -1], target)


def test_linear_styles_have_zero_curvature() -> None:
    source = torch.randn(2, 8)
    target = torch.randn(2, 8)
    time = torch.linspace(0, 1, 5)
    correction = torch.zeros(2, 5, 8)
    styles = endpoint_anchored_styles(source, target, correction, time)
    assert float(temporal_curvature_loss(styles)) < 1e-12


def test_low_rank_layer_styles_are_endpoint_anchored() -> None:
    source = torch.randn(2, 8)
    target = torch.randn(2, 8)
    time = torch.linspace(0, 1, 5)
    coefficients = torch.randn(2, 5, 3, 2)
    basis = torch.randn(2, 2, 8)
    styles = endpoint_anchored_layer_styles(
        source, target, coefficients, basis, time
    )
    assert styles.shape == (2, 5, 3, 8)
    assert torch.allclose(styles[:, 0], source[:, None].expand(-1, 3, -1))
    assert torch.allclose(styles[:, -1], target[:, None].expand(-1, 3, -1))
