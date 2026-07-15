"""Small frozen-feature probes used only for independent evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class BinaryLinearProbe:
    """Standardized logistic probe with no dependency on an image backbone."""

    mean: Tensor
    scale: Tensor
    weight: Tensor
    bias: Tensor

    def logits(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.weight.numel():
            raise ValueError("features must have shape [samples, probe_dimension]")
        standardized = (features.float() - self.mean.to(features)) / self.scale.to(features)
        return standardized @ self.weight.to(features) + self.bias.to(features)

    def probabilities(self, features: Tensor) -> Tensor:
        return self.logits(features).sigmoid()

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean.cpu(),
            "scale": self.scale.cpu(),
            "weight": self.weight.cpu(),
            "bias": self.bias.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "BinaryLinearProbe":
        required = {"mean", "scale", "weight", "bias"}
        if set(state) != required:
            raise ValueError(f"probe state must contain exactly {sorted(required)}")
        return cls(**{key: value.float().cpu() for key, value in state.items()})


def fit_binary_linear_probe(
    cat_features: Tensor,
    dog_features: Tensor,
    *,
    l2: float = 1e-3,
    max_iterations: int = 200,
) -> BinaryLinearProbe:
    """Fit class-balanced logistic regression to frozen AFHQ features.

    The class-balanced empirical risk gives cats and dogs equal total mass,
    matching the balanced endpoint measures used by the transport experiment.
    """
    if cat_features.ndim != 2 or dog_features.ndim != 2:
        raise ValueError("cat and dog features must be rank two")
    if cat_features.shape[1] != dog_features.shape[1]:
        raise ValueError("cat and dog feature dimensions differ")
    if not len(cat_features) or not len(dog_features):
        raise ValueError("both probe classes must be nonempty")
    if l2 < 0 or max_iterations <= 0:
        raise ValueError("invalid probe optimization parameters")

    cats = cat_features.detach().float().cpu()
    dogs = dog_features.detach().float().cpu()
    features = torch.cat([cats, dogs])
    labels = torch.cat([torch.zeros(len(cats)), torch.ones(len(dogs))])
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-4)
    standardized = (features - mean) / scale
    sample_weight = torch.cat(
        [torch.full((len(cats),), 0.5 / len(cats)), torch.full((len(dogs),), 0.5 / len(dogs))]
    )

    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = standardized @ weight + bias
        empirical = (
            F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            * sample_weight
        ).sum()
        objective = empirical + 0.5 * l2 * weight.square().sum()
        objective.backward()
        return objective

    optimizer.step(closure)
    return BinaryLinearProbe(
        mean=mean,
        scale=scale,
        weight=weight.detach(),
        bias=bias.detach(),
    )


def binary_probe_metrics(
    probe: BinaryLinearProbe, cat_features: Tensor, dog_features: Tensor
) -> dict[str, float]:
    """Return accuracy, balanced accuracy, and log loss on disjoint features."""
    cat_probability = probe.probabilities(cat_features).cpu()
    dog_probability = probe.probabilities(dog_features).cpu()
    cat_accuracy = float((cat_probability < 0.5).float().mean())
    dog_accuracy = float((dog_probability >= 0.5).float().mean())
    labels = torch.cat([torch.zeros_like(cat_probability), torch.ones_like(dog_probability)])
    probability = torch.cat([cat_probability, dog_probability]).clamp(1e-7, 1.0 - 1e-7)
    return {
        "accuracy": float(
            (cat_accuracy * len(cat_probability) + dog_accuracy * len(dog_probability))
            / (len(cat_probability) + len(dog_probability))
        ),
        "balanced_accuracy": 0.5 * (cat_accuracy + dog_accuracy),
        "cat_accuracy": cat_accuracy,
        "dog_accuracy": dog_accuracy,
        "log_loss": float(F.binary_cross_entropy(probability, labels)),
    }
