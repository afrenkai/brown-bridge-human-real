import torch

from brown_bridge.spatial_bridge import (
    foreground_patch_mask,
    lift_patch_trajectory,
    masked_spatial_bridge_loss,
    positional_patch_transport,
    species_clock,
)


def test_positional_patch_transport_has_expected_shapes() -> None:
    torch.manual_seed(4)
    source = torch.randn(16, 8)
    target = torch.randn(16, 8)
    mapped, plan, cost = positional_patch_transport(source, target)
    assert mapped.shape == source.shape
    assert plan.shape == (16, 16)
    assert cost.shape == (16, 16)
    assert torch.allclose(plan.sum(), torch.tensor(1.0), atol=1e-6)


def test_spatial_lift_has_exact_normalized_endpoints() -> None:
    source = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    target = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    path = torch.stack([source, 0.5 * (source + target), target])
    clock = species_clock(path.mean(1), path[-1].mean(0) - path[0].mean(0))
    lifted = lift_patch_trajectory(source, target, clock)
    assert torch.allclose(lifted[0], source, atol=1e-6)
    assert torch.allclose(lifted[-1], target, atol=1e-6)


def test_masked_loss_prefers_target_foreground_and_source_background() -> None:
    source = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    target = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    species = torch.randn(8)
    mask = foreground_patch_mask(source, target, species)
    target_path = lift_patch_trajectory(source, target, torch.tensor([0.5]))
    rendered = target_path.clone()
    loss = masked_spatial_bridge_loss(rendered, target_path, source, mask)
    assert torch.isfinite(loss)
    assert 0.05 <= float(mask.min()) <= float(mask.max()) <= 0.95
