import torch

from brown_bridge.schrodinger import EmpiricalSchrodingerBridge


def test_empirical_schrodinger_bridge_has_consistent_diffusivity_and_endpoints() -> None:
    source = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    bridge = EmpiricalSchrodingerBridge.fit(source, target, epsilon=0.4)
    assert bridge.diffusivity == 0.1
    sample = bridge.sample_paths(
        32,
        macro_steps=3,
        microsteps_per_macro=2,
        generator=torch.Generator().manual_seed(9),
    )
    assert sample.paths.shape == (7, 32, 2)
    torch.testing.assert_close(sample.paths[0], sample.endpoints.source.double())
    torch.testing.assert_close(sample.paths[-1], sample.endpoints.target.double())
    assert bridge.conditional_marginal_variance(0.5) == 0.05
