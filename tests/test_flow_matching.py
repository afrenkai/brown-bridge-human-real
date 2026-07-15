import torch
from torch import nn

from brown_bridge.flow_matching import integrate_flow, make_ot_cfm_batch


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
