import torch

from brown_bridge.layerwise_style import (
    EndpointVanishingResidualStyleController,
    MonotoneLayerwiseStyleController,
    chroma_laplacian,
    clip_layerwise_clock,
    layerwise_clock_regularizer,
    layerwise_support_regularizer,
    localized_chroma_artifact_loss,
    residual_style_regularizers,
)


def test_layerwise_controller_is_endpoint_pinned_and_monotone() -> None:
    controller = MonotoneLayerwiseStyleController(5, 7, 3, hidden_dimension=11)
    source = torch.randn(4, 5)
    target = torch.randn(4, 5)
    context = torch.randn(4, 7)
    times = torch.linspace(0, 1, 9)
    styles, clock = controller(source, target, context, times)
    assert styles.shape == (4, 9, 3, 5)
    assert clock.shape == (4, 9, 3)
    torch.testing.assert_close(styles[:, 0], source[:, None].expand(-1, 3, -1))
    torch.testing.assert_close(styles[:, -1], target[:, None].expand(-1, 3, -1))
    assert torch.all(clock[:, 1:] >= clock[:, :-1])


def test_zero_initialized_controller_is_shared_identity_clock() -> None:
    controller = MonotoneLayerwiseStyleController(2, 3, 4, hidden_dimension=5)
    times = torch.linspace(0, 1, 7)
    clock = controller.clock(torch.randn(6, 3), times)
    torch.testing.assert_close(clock, times[None, :, None].expand(6, -1, 4))
    assert float(layerwise_clock_regularizer(clock, times).detach()) < 1e-12


def test_layerwise_controller_validates_inputs_and_backpropagates() -> None:
    controller = MonotoneLayerwiseStyleController(3, 4, 2)
    source = torch.randn(2, 3)
    target = torch.randn(2, 3)
    context = torch.randn(2, 4)
    styles, _ = controller(source, target, context, torch.tensor([0.0, 0.4, 1.0]))
    styles.square().mean().backward()
    assert controller.base_shift.grad is not None
    try:
        controller.clock(context, torch.tensor([0.0, 1.1]))
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("invalid time grid was accepted")


def test_support_tube_clips_layer_disagreement_and_preserves_endpoints() -> None:
    clock = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.5, 0.9], [1.0, 1.0, 1.0]]]
    )
    clipped = clip_layerwise_clock(clock, 0.2)
    torch.testing.assert_close(clipped[:, 0], clock[:, 0])
    torch.testing.assert_close(clipped[:, -1], clock[:, -1])
    assert torch.all((clipped - clipped.mean(2, keepdim=True)).abs() <= 0.2 + 1e-6)
    assert float(layerwise_support_regularizer(clipped, 0.2)) == 0.0
    assert float(layerwise_support_regularizer(clock, 0.2)) > 0.0


def test_localized_chroma_loss_detects_single_colored_spike() -> None:
    endpoints = torch.zeros(2, 2, 3, 12, 12)
    clean = torch.zeros(2, 3, 3, 12, 12, requires_grad=True)
    corrupted = clean.clone()
    corrupted[0, 1, :, 6, 6] = torch.tensor([1.0, 0.0, 1.0])
    assert chroma_laplacian(torch.zeros(1, 3, 4, 4)).shape == (1, 2, 2)
    clean_loss = localized_chroma_artifact_loss(clean, endpoints)
    corrupted_loss = localized_chroma_artifact_loss(corrupted, endpoints)
    assert float(clean_loss.detach()) == 0.0
    assert float(corrupted_loss.detach()) > 0.0
    clean_loss.backward()
    assert torch.isfinite(clean.grad).all()


def test_residual_controller_is_low_rank_bounded_and_endpoint_pinned() -> None:
    controller = EndpointVanishingResidualStyleController(
        5, 7, 3, hidden_dimension=11, residual_rank=2, max_residual_ratio=0.15
    )
    source = torch.randn(4, 5)
    target = torch.randn(4, 5)
    context = torch.randn(4, 7)
    times = torch.linspace(0, 1, 9)
    styles, clock, residual = controller(source, target, context, times)
    assert styles.shape == residual.shape == (4, 9, 3, 5)
    assert clock.shape == (4, 9, 3)
    torch.testing.assert_close(residual[:, 0], torch.zeros_like(residual[:, 0]))
    torch.testing.assert_close(residual[:, -1], torch.zeros_like(residual[:, -1]))
    torch.testing.assert_close(styles[:, 0], source[:, None].expand(-1, 3, -1))
    torch.testing.assert_close(styles[:, -1], target[:, None].expand(-1, 3, -1))
    loss = styles[:, 1:-1].square().mean()
    loss.backward()
    final = controller.residual_net[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()


def test_residual_regularizers_measure_energy_and_curvature() -> None:
    source = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    linear = torch.linspace(0, 1, 5)[None, :, None, None].expand(2, -1, 3, 4)
    energy, curvature = residual_style_regularizers(linear, source, target)
    assert float(energy) > 0
    assert float(curvature) < 1e-12
    bumped = linear.clone()
    bumped[:, 2] += 0.5
    _, bumped_curvature = residual_style_regularizers(bumped, source, target)
    assert float(bumped_curvature) > 0
