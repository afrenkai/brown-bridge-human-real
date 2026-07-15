import torch

from brown_bridge.probes import (
    BinaryLinearProbe,
    binary_probe_metrics,
    fit_binary_linear_probe,
)


def test_binary_probe_fits_and_round_trips() -> None:
    generator = torch.Generator().manual_seed(17)
    cats = -1.5 + 0.25 * torch.randn(80, 5, generator=generator)
    dogs = 1.5 + 0.25 * torch.randn(65, 5, generator=generator)
    probe = fit_binary_linear_probe(cats, dogs, max_iterations=50)
    restored = BinaryLinearProbe.from_state_dict(probe.state_dict())

    assert torch.allclose(probe.probabilities(cats), restored.probabilities(cats))
    metrics = binary_probe_metrics(restored, cats, dogs)
    assert metrics["balanced_accuracy"] > 0.99
    assert metrics["log_loss"] < 0.05


def test_binary_probe_rejects_wrong_dimension() -> None:
    probe = BinaryLinearProbe(
        mean=torch.zeros(3),
        scale=torch.ones(3),
        weight=torch.ones(3),
        bias=torch.tensor(0.0),
    )
    try:
        probe.probabilities(torch.zeros(2, 4))
    except ValueError as error:
        assert "probe_dimension" in str(error)
    else:
        raise AssertionError("dimension mismatch should fail")
