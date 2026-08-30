import torch

from brown_bridge.paths import sample_coupling


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
    assert sampled.source.shape == sampled.target.shape == (1_000, 1)


def test_sampling_rejects_plan_shape_mismatch() -> None:
    plan = torch.ones(2, 3)
    source = torch.ones(2, 1)
    target = torch.ones(2, 1)
    try:
        sample_coupling(plan, source, target, 1)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("shape mismatch was accepted")
