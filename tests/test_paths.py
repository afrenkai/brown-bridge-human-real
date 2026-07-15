import torch

from brown_bridge.paths import (
    CoupledEndpoints,
    brownian_bridge,
    interpolate,
    make_time_grid,
    sample_coupling,
)


def endpoints(source: torch.Tensor, target: torch.Tensor) -> CoupledEndpoints:
    count = len(source)
    indices = torch.arange(count)
    return CoupledEndpoints(source, target, indices, indices)


def test_time_grid_has_macro_boundaries_and_microsteps() -> None:
    times = make_time_grid(3, 4)
    assert len(times) == 13
    torch.testing.assert_close(
        times[[0, 4, 8, 12]],
        torch.tensor([0, 1 / 3, 2 / 3, 1], dtype=times.dtype),
    )
    torch.testing.assert_close(
        times[1:] - times[:-1], torch.full((12,), 1 / 12, dtype=times.dtype)
    )


def test_spherical_path_stays_normalized_and_hits_endpoints() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    times = torch.linspace(0, 1, 9)
    trajectory, velocity = interpolate(
        endpoints(source, target), times, geometry="spherical", return_velocity=True
    )
    torch.testing.assert_close(trajectory[0], source.double())
    torch.testing.assert_close(trajectory[-1], target.double(), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        torch.linalg.vector_norm(trajectory, dim=2),
        torch.ones((9, 2), dtype=torch.float64),
    )
    # Great-circle velocity is tangent to the sphere.
    torch.testing.assert_close(
        (trajectory * velocity).sum(2),
        torch.zeros((9, 2), dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


def test_spherical_path_handles_antipodes() -> None:
    source = torch.tensor([[1.0, 0.0, 0.0]])
    target = -source
    trajectory = interpolate(
        endpoints(source, target), torch.tensor([0.0, 0.5, 1.0]), geometry="spherical"
    )
    torch.testing.assert_close(trajectory[0], source.double())
    torch.testing.assert_close(trajectory[-1], target.double(), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(trajectory, dim=2),
        torch.ones((3, 1), dtype=torch.float64),
    )


def test_brownian_bridge_is_pinned_at_both_ends() -> None:
    source = torch.zeros((4, 3))
    target = torch.ones((4, 3))
    times = make_time_grid(2, 5)
    trajectory = brownian_bridge(
        endpoints(source, target),
        times,
        diffusivity=0.2,
        generator=torch.Generator().manual_seed(3),
    )
    torch.testing.assert_close(trajectory[0], source.double())
    torch.testing.assert_close(trajectory[-1], target.double())
    assert not torch.allclose(trajectory[5], torch.full((4, 3), 0.5, dtype=torch.float64))


def test_sampling_uses_joint_plan() -> None:
    source = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[10.0], [20.0]])
    plan = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    sampled = sample_coupling(
        plan,
        source,
        target,
        1_000,
        generator=torch.Generator().manual_seed(4),
    )
    assert torch.all(sampled.source_indices == sampled.target_indices)
