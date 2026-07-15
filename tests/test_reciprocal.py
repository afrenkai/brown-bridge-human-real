import torch

from brown_bridge.reciprocal import (
    DirectionalTimeMLP,
    InequalityAugmentedLagrangian,
    integrate_directional_sde,
    orthogonal_displacement,
    species_progression_loss,
)


def test_augmented_lagrangian_dual_projection_and_primal_gradient() -> None:
    constraint = InequalityAugmentedLagrangian(threshold=0.2, penalty=10.0)
    feasible = torch.tensor(0.1, requires_grad=True)
    assert torch.isclose(constraint.primal_loss(feasible), torch.tensor(0.0))
    assert constraint.dual_step(feasible) == 0.0

    violated = torch.tensor(0.3, requires_grad=True)
    loss = constraint.primal_loss(violated)
    loss.backward()
    assert float(violated.grad) > 0
    assert abs(constraint.dual_step(violated) - 1.0) < 1e-6


def test_orthogonal_displacement_ignores_species_axis() -> None:
    reference = torch.zeros(2, 3)
    species = torch.tensor([1.0, 0.0, 0.0])
    along_species = torch.tensor([[3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    value = orthogonal_displacement(along_species, reference, species)
    assert torch.isclose(value, torch.tensor(0.0))


def test_species_progression_prefers_correct_signed_velocity() -> None:
    species = torch.tensor([1.0, 0.0])
    directions = torch.tensor([1, -1])
    correct = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    wrong = -correct
    assert species_progression_loss(correct, directions, species) < species_progression_loss(
        wrong, directions, species
    )


def test_zero_field_zero_noise_is_constant() -> None:
    velocity = DirectionalTimeMLP(2, width=8, depth=1, time_frequencies=1)
    score = DirectionalTimeMLP(2, width=8, depth=1, time_frequencies=1)
    for model in (velocity, score):
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
    initial = torch.tensor([[1.0, -1.0]])
    noise = torch.zeros(4, 1, 2)
    path = integrate_directional_sde(
        velocity, score, initial, direction=1, steps=4, kappa=0.1, noise=noise
    )
    assert torch.allclose(path, initial.expand_as(path))
