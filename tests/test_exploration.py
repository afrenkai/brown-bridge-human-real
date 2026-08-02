import pytest
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


def test_best_of_k_supports_unreduced_and_sum_reductions() -> None:
    candidate_losses = torch.tensor([[3.0, 1.0], [4.0, 2.0]])

    none_result = best_of_k_loss(candidate_losses, reduction="none")
    sum_result = best_of_k_loss(candidate_losses, reduction="sum")

    torch.testing.assert_close(none_result.loss, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(sum_result.loss, torch.tensor(3.0))


@pytest.mark.parametrize(
    "candidate_losses, message",
    [
        (torch.ones(3), "shape"),
        (torch.ones(2, 0), "at least one"),
        (torch.ones(2, 2, dtype=torch.int64), "floating-point"),
        (torch.tensor([[1.0, float("nan")]]), "finite"),
    ],
)
def test_best_of_k_rejects_invalid_losses(
    candidate_losses: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        best_of_k_loss(candidate_losses)
