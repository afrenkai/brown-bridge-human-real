import pytest
import torch
from torch import nn

from brown_bridge.flow_matching import (
    SpatialTimeConditionedField,
    integrate_flow,
    make_ot_cfm_batch,
)


class ConstantField(nn.Module):
    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(state)


def test_ambient_integrator_uses_all_microsteps() -> None:
    initial = torch.zeros((3, 2))
    times, trajectory = integrate_flow(
        ConstantField(),
        initial,
        macro_steps=5,
        microsteps_per_macro=3,
        method="rk4",
        geometry="ambient",
    )
    assert len(times) == len(trajectory) == 16
    torch.testing.assert_close(trajectory[-1], torch.ones_like(initial))


def test_ot_cfm_spherical_targets_are_tangent() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    plan = torch.eye(2) / 2
    batch = make_ot_cfm_batch(
        plan,
        source,
        target,
        64,
        geometry="spherical",
        generator=torch.Generator().manual_seed(5),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(batch.state, dim=1), torch.ones(64)
    )
    torch.testing.assert_close(
        (batch.state * batch.conditional_velocity).sum(1),
        torch.zeros(64),
        atol=1e-6,
        rtol=0,
    )


def test_spatial_time_field_preserves_flattened_coordinate_shape() -> None:
    field = SpatialTimeConditionedField(
        latent_channels=2, spatial_size=8, base_channels=8, time_dim=16
    )
    state = torch.randn(3, 2 * 8 * 8, requires_grad=True)
    velocity = field(state, torch.tensor([0.0, 0.5, 1.0]))
    assert velocity.shape == state.shape
    velocity.square().mean().backward()
    assert state.grad is not None


def test_spatial_time_field_rejects_nonspatial_coordinate() -> None:
    field = SpatialTimeConditionedField(
        latent_channels=2, spatial_size=8, base_channels=8, time_dim=16
    )
    with pytest.raises(ValueError, match="state must have shape"):
        field(torch.randn(3, 17), torch.rand(3))


def test_spatial_transformer_field_and_reverse_integration() -> None:
    field = SpatialTimeConditionedField(
        2,
        16,
        base_channels=8,
        time_dim=16,
        transformer_blocks=1,
        transformer_heads=4,
    )
    initial = torch.randn(2, 2 * 16 * 16)
    times, path = integrate_flow(
        field,
        initial,
        macro_steps=2,
        microsteps_per_macro=1,
        geometry="ambient",
        reverse=True,
    )
    assert times.tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert path.shape == (3, 2, 2 * 16 * 16)


def test_spatial_field_supports_joint_velocity_score_output() -> None:
    field = SpatialTimeConditionedField(
        4,
        8,
        base_channels=8,
        time_dim=16,
        transformer_blocks=1,
        transformer_heads=4,
        output_channels=8,
    )
    state = torch.randn(3, 4 * 8 * 8)
    output = field(state, torch.tensor([0.1, 0.5, 0.9]))
    assert output.shape == (3, 8 * 8 * 8)
