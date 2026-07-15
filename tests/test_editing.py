import torch
from torch import nn

from brown_bridge.editing import (
    flow_edit,
    integrate_residual_edit,
    sphere_exp_map,
    sphere_log_map,
)


def test_spherical_log_exp_round_trip() -> None:
    anchor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    reconstructed = sphere_exp_map(anchor, sphere_log_map(anchor, target))
    torch.testing.assert_close(reconstructed, target, atol=1e-6, rtol=1e-6)


class TangentResidualField(nn.Module):
    def forward(
        self,
        residual: torch.Tensor,
        time: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(residual)
        output[:, 1] = torch.pi / 2
        return output


def test_residual_edit_is_source_anchored() -> None:
    source = torch.tensor([[1.0, 0.0]])
    times, residuals, features = integrate_residual_edit(
        TangentResidualField(),
        source,
        macro_steps=2,
        microsteps_per_macro=4,
        geometry="spherical",
    )
    assert len(times) == len(residuals) == len(features) == 9
    torch.testing.assert_close(features[0], source)
    torch.testing.assert_close(
        features[-1], torch.tensor([[0.0, 1.0]]), atol=1e-5, rtol=0
    )


class ZeroField(nn.Module):
    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state)


class OneField(nn.Module):
    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(state)


def test_flow_edit_integrates_velocity_difference_backward() -> None:
    source = torch.zeros((2, 3))
    times, path = flow_edit(
        source,
        ZeroField(),
        OneField(),
        macro_steps=2,
        microsteps_per_macro=5,
        averages=3,
        generator=torch.Generator().manual_seed(7),
    )
    assert times[0] == 1 and times[-1] == 0
    torch.testing.assert_close(path[-1], -torch.ones_like(source))
