import torch

from brown_bridge.light_sb import (
    CoordinateStandardizer,
    DiagonalLightSB,
    brownian_bridge_path,
    coarse_latent_coordinate,
    expected_minibatch_ot_pairs,
    lift_coarse_coordinate,
)


def test_standardizer_round_trip() -> None:
    samples = torch.randn(23, 7) * torch.arange(1, 8)
    standardizer = CoordinateStandardizer.fit(samples)
    recovered = standardizer.inverse(standardizer.transform(samples))
    assert torch.allclose(recovered, samples, atol=1e-6)


def test_coarse_lift_preserves_residual_and_sets_coordinate() -> None:
    source = torch.randn(3, 2, 8, 8)
    coordinate = torch.randn(3, 2 * 4 * 4)
    lifted = lift_coarse_coordinate(source, coordinate, factor=2)
    assert torch.allclose(coarse_latent_coordinate(lifted, factor=2), coordinate, atol=1e-6)
    assert torch.allclose(
        lift_coarse_coordinate(source, coarse_latent_coordinate(source, factor=2), factor=2),
        source,
        atol=1e-6,
    )
    contracted = lift_coarse_coordinate(
        source, coordinate, factor=2, residual_weight=torch.zeros(3, 1, 1, 1)
    )
    assert torch.allclose(
        contracted,
        torch.nn.functional.interpolate(
            coordinate.reshape(3, 2, 4, 4), scale_factor=2, mode="nearest"
        ),
    )


def test_lightsb_objective_has_gradients_and_conditional_shapes() -> None:
    model = DiagonalLightSB(5, components=3, epsilon=0.2)
    source = torch.randn(11, 5)
    target = torch.randn(13, 5) + 1
    loss = model.objective(source, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.conditional_mean(source).shape == source.shape
    assert model.sample_endpoint(source, generator=torch.Generator().manual_seed(3)).shape == source.shape


def test_one_component_conditional_mean_matches_affine_formula() -> None:
    model = DiagonalLightSB(2, components=1, epsilon=0.1)
    with torch.no_grad():
        model.centers.copy_(torch.tensor([[2.0, -1.0]]))
        model.log_scale.copy_(torch.tensor([[0.0, torch.log(torch.tensor(2.0))]]))
    source = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(model.conditional_mean(source), torch.tensor([[5.0, 7.0]]))


def test_brownian_bridge_is_endpoint_anchored_and_reproducible() -> None:
    source = torch.zeros(4, 3)
    target = torch.ones(4, 3)
    times = torch.linspace(0, 1, 9)
    first = brownian_bridge_path(
        source, target, times, epsilon=0.2, generator=torch.Generator().manual_seed(8)
    )
    second = brownian_bridge_path(
        source, target, times, epsilon=0.2, generator=torch.Generator().manual_seed(8)
    )
    assert torch.equal(first, second)
    assert torch.equal(first[0], source)
    assert torch.equal(first[-1], target)


def test_expected_minibatch_ot_pairs_follow_separated_modes() -> None:
    source = torch.cat([torch.randn(40, 2) * 0.01 - 2, torch.randn(40, 2) * 0.01 + 2])
    target = torch.cat([torch.randn(40, 2) * 0.01 - 3, torch.randn(40, 2) * 0.01 + 3])
    paired_source, paired_target = expected_minibatch_ot_pairs(
        source,
        target,
        batch_size=32,
        batches=8,
        epsilon=0.02,
        generator=torch.Generator().manual_seed(2),
    )
    assert paired_source.shape == paired_target.shape == (256, 2)
    assert float((paired_source[:, 0] * paired_target[:, 0] > 0).float().mean()) > 0.95


def test_expected_minibatch_ot_pairs_can_use_external_cost_features() -> None:
    source = torch.cat([-torch.ones(40, 1), torch.ones(40, 1)])
    target = source.clone()
    source_feature = torch.cat(
        [torch.tensor([[-1.0, 0.0]]).repeat(40, 1), torch.tensor([[1.0, 0.0]]).repeat(40, 1)]
    )
    target_feature = -source_feature
    paired_source, paired_target = expected_minibatch_ot_pairs(
        source,
        target,
        batch_size=64,
        batches=1,
        epsilon=0.02,
        generator=torch.Generator().manual_seed(4),
        source_cost_features=source_feature,
        target_cost_features=target_feature,
        pairing="rounded",
    )
    assert float((paired_source[:, 0] * paired_target[:, 0] < 0).float().mean()) > 0.95
