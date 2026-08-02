import torch
from torch import nn

from brown_bridge.spatial_stochastic import (
    AnchoredSpatialVelocityScoreField,
    SpatialVelocityScoreField,
    bridge_diffusion_rate,
    bridge_score_scale,
    integrate_anchored_probability_flow,
    integrate_anchored_sde,
    integrate_probability_flow,
    integrate_reciprocal_sde,
    sample_pinned_brownian_bridge,
)


class ConstantVelocityScore(nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones_like(state), torch.zeros_like(state)


class ConstantAnchoredVelocityScore(nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        anchor: torch.Tensor,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return direction[:, None] * torch.ones_like(state), torch.zeros_like(state)


def test_joint_uvit_field_shape_and_gradient() -> None:
    field = SpatialVelocityScoreField(
        2,
        8,
        base_channels=8,
        time_dim=16,
        transformer_blocks=1,
        transformer_heads=4,
    )
    state = torch.randn(3, 2 * 8 * 8, requires_grad=True)
    velocity, scaled_score = field(state, torch.tensor([0.1, 0.5, 0.9]))
    assert velocity.shape == scaled_score.shape == state.shape
    (velocity.square().mean() + scaled_score.square().mean()).backward()
    assert state.grad is not None


def test_anchored_uvit_field_shape_and_gradient() -> None:
    field = AnchoredSpatialVelocityScoreField(
        2,
        8,
        base_channels=8,
        time_dim=16,
        transformer_blocks=1,
        transformer_heads=4,
    )
    state = torch.randn(3, 2 * 8 * 8, requires_grad=True)
    anchor = torch.randn_like(state)
    velocity, scaled_score = field(
        state,
        torch.tensor([0.1, 0.5, 0.9]),
        anchor,
        torch.tensor([1.0, -1.0, 1.0]),
    )
    assert velocity.shape == scaled_score.shape == state.shape
    (velocity.square().mean() + scaled_score.square().mean()).backward()
    assert state.grad is not None


def test_bridge_rate_and_score_scale_have_expected_geometry() -> None:
    time = torch.tensor([0.0, 0.5, 1.0])
    rate = bridge_diffusion_rate(time, peak_rate=0.2)
    torch.testing.assert_close(rate, torch.tensor([0.0, 0.2, 0.0]))
    scale = bridge_score_scale(time, interpolant_epsilon=0.08, minimum_time=0.01)
    assert scale[1] > scale[0]
    torch.testing.assert_close(scale[0], scale[2])


def test_forward_and_reverse_probability_flow_use_physical_time_direction() -> None:
    field = ConstantVelocityScore()
    initial = torch.zeros(2, 3)
    forward = integrate_probability_flow(field, initial, macro_steps=4)
    reverse = integrate_probability_flow(field, initial, macro_steps=4, reverse=True)
    torch.testing.assert_close(forward[-1], torch.ones_like(initial))
    torch.testing.assert_close(reverse[-1], -torch.ones_like(initial))


def test_zero_rate_reciprocal_sde_reduces_to_current_flow() -> None:
    field = ConstantVelocityScore()
    initial = torch.zeros(2, 3)
    noise = torch.randn(4, 2, 3)
    path = integrate_reciprocal_sde(
        field,
        initial,
        interpolant_epsilon=0.08,
        peak_diffusion_rate=0.0,
        macro_steps=4,
        noise=noise,
    )
    torch.testing.assert_close(path[-1], torch.ones_like(initial))


def test_anchored_integrators_share_direction_condition() -> None:
    field = ConstantAnchoredVelocityScore()
    initial = torch.zeros(2, 3)
    forward = integrate_anchored_probability_flow(
        field, initial, direction=1, macro_steps=4
    )
    reverse = integrate_anchored_sde(
        field,
        initial,
        direction=-1,
        interpolant_epsilon=0.08,
        peak_diffusion_rate=0.0,
        macro_steps=4,
    )
    torch.testing.assert_close(forward[-1], torch.ones_like(initial))
    torch.testing.assert_close(reverse[-1], -torch.ones_like(initial))


def test_pinned_brownian_bridge_has_exact_endpoints_and_linear_zero_noise_mean() -> None:
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
    noisy = sample_pinned_brownian_bridge(
        initial,
        target,
        epsilon=0.1,
        macro_steps=4,
    )
    torch.testing.assert_close(noisy[0], initial)
    torch.testing.assert_close(noisy[-1], target)
