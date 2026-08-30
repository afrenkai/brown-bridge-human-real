import torch

from brown_bridge.features import (
    compact_spatial_content_descriptor,
    content_cost_matrix,
    project_off_direction,
    species_direction,
)


def test_content_descriptor_keeps_coarse_spatial_layout() -> None:
    global_features = torch.randn(3, 8)
    patch_features = torch.randn(3, 16, 8)
    direction = species_direction(torch.randn(4, 8), torch.randn(5, 8) + 1.0)
    descriptor = compact_spatial_content_descriptor(
        global_features, patch_features, direction
    )
    assert descriptor.shape == (3, 8 + 4 * 8)
    assert torch.isfinite(descriptor).all()
    projected = project_off_direction(global_features, direction)
    torch.testing.assert_close(
        projected @ direction,
        torch.zeros(3),
        atol=1e-6,
        rtol=0,
    )


def test_content_cost_matrix_has_exact_diagonal() -> None:
    descriptors = torch.randn(5, 7)
    costs = content_cost_matrix(descriptors, descriptors)
    assert costs.shape == (5, 5)
    torch.testing.assert_close(costs.diag(), torch.zeros(5, dtype=torch.float64))
