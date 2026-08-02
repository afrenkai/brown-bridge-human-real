import torch

from brown_bridge.terminal_distillation import (
    sample_terminal_indices,
    topk_terminal_support,
)


def test_topk_terminal_support_combines_semantic_and_layout_costs() -> None:
    source_descriptor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_descriptor = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    source_layout = source_descriptor.clone()
    target_layout = target_descriptor.clone()
    indices, costs = topk_terminal_support(
        source_descriptor,
        target_descriptor,
        source_layout,
        target_layout,
        neighbors=2,
        layout_weight=0.3,
        chunk_size=1,
    )
    assert indices.shape == costs.shape == (2, 2)
    assert indices[:, 0].tolist() == [0, 1]
    torch.testing.assert_close(costs[:, 0], torch.zeros(2))


def test_terminal_sampling_supports_deterministic_and_gibbs_modes() -> None:
    indices = torch.tensor([[4, 5], [7, 8]])
    costs = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
    assert sample_terminal_indices(indices, costs, temperature=0.0).tolist() == [4, 7]
    sampled = sample_terminal_indices(
        indices,
        costs,
        temperature=0.01,
        generator=torch.Generator().manual_seed(3),
    )
    assert sampled.tolist() == [4, 7]
