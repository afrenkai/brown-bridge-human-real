import torch

from brown_bridge.continuous_vae import (
    ContinuousConditionalVAE,
    diagonal_gaussian_kl,
    latent_brownian_bridge,
    sinusoidal_time_embedding,
)


def test_continuous_vae_shapes() -> None:
    model = ContinuousConditionalVAE(base_channels=8, latent_channels=4, condition_dim=16)
    image = torch.rand(2, 3, 128, 128)
    time = torch.tensor([0.0, 1.0])
    global_state = torch.randn(2, 192)
    patches = torch.randn(2, 16, 192)
    output, mean, log_variance = model(image, time, global_state, patches)
    assert output.shape == image.shape
    assert mean.shape == log_variance.shape == (2, 4, 4, 4)
    assert torch.isfinite(diagonal_gaussian_kl(mean, log_variance))


def test_latent_bridge_is_endpoint_pinned() -> None:
    source = torch.randn(2, 3, 4, 4)
    target = torch.randn_like(source)
    noise = torch.randn_like(source)
    path = latent_brownian_bridge(
        source, target, torch.tensor([0.0, 1.0]), diffusivity=0.2, noise=noise
    )
    torch.testing.assert_close(path[0], source[0])
    torch.testing.assert_close(path[1], target[1])


def test_time_embedding_shape_and_origin() -> None:
    embedding = sinusoidal_time_embedding(torch.tensor([0.0, 0.5]), 8)
    assert embedding.shape == (2, 8)
    torch.testing.assert_close(embedding[0, :4], torch.zeros(4))
    torch.testing.assert_close(embedding[0, 4:], torch.ones(4))


def test_eight_by_eight_latent_grid() -> None:
    model = ContinuousConditionalVAE(
        base_channels=8,
        latent_channels=4,
        condition_dim=16,
        latent_grid=8,
    )
    image = torch.rand(1, 3, 128, 128)
    mean, log_variance = model.encode(image)
    assert mean.shape == log_variance.shape == (1, 4, 8, 8)
    output = model.decode(
        mean,
        torch.tensor([0.5]),
        torch.randn(1, 192),
        torch.randn(1, 16, 192),
    )
    assert output.shape == image.shape
