import torch

from brown_bridge.exploration import best_of_k_loss


def test_best_of_k_selects_one_candidate_per_example() -> None:
    candidate_losses = torch.tensor(
        [[3.0, 1.0, 2.0], [0.5, 0.75, 0.25]], requires_grad=True
    )
    result = best_of_k_loss(candidate_losses)
    torch.testing.assert_close(result.loss, torch.tensor(0.625))
    torch.testing.assert_close(result.winner_indices, torch.tensor([1, 2]))
    result.loss.backward()
    torch.testing.assert_close(
        candidate_losses.grad,
        torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
    )


def test_best_of_k_reductions() -> None:
    candidate_losses = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
    torch.testing.assert_close(
        best_of_k_loss(candidate_losses, reduction="none").loss,
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        best_of_k_loss(candidate_losses, reduction="sum").loss,
        torch.tensor(3.0),
    )
