import torch

from brown_bridge.ot import pairwise_cost, sinkhorn, sinkhorn_from_cost


def test_sinkhorn_respects_nonuniform_marginals() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    source_mass = torch.tensor([0.3, 0.7])
    target_mass = torch.tensor([0.2, 0.5, 0.3])
    result = sinkhorn(
        source,
        target,
        source_weights=source_mass,
        target_weights=target_mass,
        epsilon=0.2,
        tolerance=1e-10,
        max_iterations=2_000,
    )
    assert result.converged
    torch.testing.assert_close(
        result.plan.sum(1), source_mass.double(), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        result.plan.sum(0), target_mass.double(), atol=1e-8, rtol=0
    )


def test_precomputed_cost_matches_feature_entry_point() -> None:
    source = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[0.25], [1.25]])
    direct = sinkhorn(source, target, epsilon=0.2, cost="sqeuclidean")
    precomputed = sinkhorn_from_cost(
        pairwise_cost(source, target, "sqeuclidean"), epsilon=0.2
    )
    torch.testing.assert_close(direct.plan, precomputed.plan)


def test_pairwise_cost_supports_all_retained_geometries() -> None:
    source = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])
    torch.testing.assert_close(
        pairwise_cost(source, target, "sqeuclidean"),
        torch.tensor([[2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        pairwise_cost(source, target, "cosine"),
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        pairwise_cost(source, target, "angular_squared"),
        torch.tensor([[torch.pi**2 / 4]], dtype=torch.float64),
    )
