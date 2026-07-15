import torch

from brown_bridge.refiner import SDEConditionedRefiner, deformation_temporal_loss


def test_refiner_is_identity_at_endpoints() -> None:
    model = SDEConditionedRefiner(feature_dim=8, condition_dim=16)
    base = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    source = torch.randn_like(base)
    state = torch.randn(2, 8)
    time = torch.tensor([0.0, 1.0])
    output, flow, residual = model(base, source, state, state, state, time)
    assert torch.allclose(output, base, atol=1e-6)
    assert torch.count_nonzero(flow) == 0
    assert torch.count_nonzero(residual) == 0


def test_deformation_temporal_loss_is_zero_for_zero_path() -> None:
    flow = torch.zeros(2, 5, 2, 8, 8)
    residual = torch.zeros(2, 5, 3, 8, 8)
    assert float(deformation_temporal_loss(flow, residual)) == 0.0


def test_band_limited_flow_only_refiner_preserves_shape_and_endpoints() -> None:
    model = SDEConditionedRefiner(
        feature_dim=8,
        condition_dim=16,
        max_residual=0.0,
        output_grid_size=4,
    )
    base = torch.randn(3, 3, 16, 16).clamp(-1, 1)
    source = torch.randn_like(base).clamp(-1, 1)
    start = torch.randn(3, 8)
    bridge = torch.randn(3, 8)
    target = torch.randn(3, 8)
    time = torch.tensor([0.0, 0.5, 1.0])
    output, flow, residual = model(base, source, start, bridge, target, time)
    assert output.shape == base.shape
    assert flow.shape == (3, 2, 16, 16)
    assert torch.count_nonzero(residual) == 0
    torch.testing.assert_close(output[0], base[0])
    torch.testing.assert_close(output[-1], base[-1])
