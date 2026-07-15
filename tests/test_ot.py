import torch

from brown_bridge.ot import barycentric_projection, sinkhorn, sinkhorn_divergence


def test_sinkhorn_respects_nonuniform_marginals() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    a = torch.tensor([0.3, 0.7])
    b = torch.tensor([0.2, 0.5, 0.3])
    result = sinkhorn(
        source,
        target,
        source_weights=a,
        target_weights=b,
        epsilon=0.2,
        tolerance=1e-10,
        max_iterations=2_000,
    )
    assert result.converged
    expected_a = a.double() / a.double().sum()
    expected_b = b.double() / b.double().sum()
    torch.testing.assert_close(result.plan.sum(1), expected_a, atol=1e-9, rtol=0)
    torch.testing.assert_close(result.plan.sum(0), expected_b, atol=1e-9, rtol=0)


def test_sinkhorn_divergence_is_zero_on_identical_measure() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    divergence, _ = sinkhorn_divergence(
        features,
        features,
        epsilon=0.1,
        tolerance=1e-10,
        max_iterations=2_000,
    )
    assert abs(float(divergence)) < 1e-10


def test_sinkhorn_divergence_accepts_different_nonuniform_sample_counts() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    divergence, cross = sinkhorn_divergence(
        source,
        target,
        source_weights=torch.tensor([0.4, 0.6]),
        target_weights=torch.tensor([0.2, 0.5, 0.3]),
        epsilon=0.2,
    )
    assert torch.isfinite(divergence)
    assert cross.plan.shape == (2, 3)


def test_barycentric_projection_is_marked_as_a_summary_by_behavior() -> None:
    # A diffuse one-to-two conditional collapses both targets to their mean.
    source = torch.tensor([[0.0]])
    target = torch.tensor([[-1.0], [1.0]])
    plan = torch.tensor([[0.5, 0.5]])
    projected = barycentric_projection(plan, source, target)
    torch.testing.assert_close(projected, torch.zeros_like(projected))
