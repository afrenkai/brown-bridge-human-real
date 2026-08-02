import pytest
import torch

from brown_bridge.latent_diffusion import (
    BidirectionalLatentDenoiser,
    conditional_latent_ddim_sample,
    q_sample,
)
from brown_bridge.residual_diffusion import cosine_alpha_bar


def tiny_model() -> BidirectionalLatentDenoiser:
    return BidirectionalLatentDenoiser(2, base_channels=8, time_dim=16, latent_size=8)


def test_bidirectional_denoiser_shape_and_gradient() -> None:
    model = tiny_model()
    target = torch.randn(3, 2, 8, 8, requires_grad=True)
    prediction = model(target, torch.randn_like(target), torch.rand(3), torch.ones(3))
    assert prediction.shape == target.shape
    prediction.square().mean().backward()
    assert target.grad is not None


def test_transformer_bottleneck_preserves_spatial_shape() -> None:
    model = BidirectionalLatentDenoiser(
        2,
        base_channels=8,
        time_dim=16,
        latent_size=16,
        transformer_blocks=1,
        transformer_heads=4,
    )
    target = torch.randn(2, 2, 16, 16)
    assert model(target, torch.randn_like(target), torch.rand(2), torch.ones(2)).shape == target.shape


def test_transformer_bottleneck_rejects_wrong_configured_size() -> None:
    model = BidirectionalLatentDenoiser(
        2,
        base_channels=8,
        time_dim=16,
        latent_size=16,
        transformer_blocks=1,
        transformer_heads=4,
    )
    target = torch.randn(2, 2, 32, 32)
    with pytest.raises(ValueError, match="configured latent_size"):
        model(target, torch.randn_like(target), torch.rand(2), torch.ones(2))


def test_q_sample_keeps_shape() -> None:
    clean = torch.randn(4, 2, 8, 8)
    sample = q_sample(clean, torch.randn_like(clean), torch.arange(4), cosine_alpha_bar(8))
    assert sample.shape == clean.shape


def test_conditional_sampler_supports_both_directions() -> None:
    model = tiny_model().eval()
    condition = torch.randn(2, 2, 8, 8)
    noise = torch.randn_like(condition)
    schedule = cosine_alpha_bar(8)
    forward = conditional_latent_ddim_sample(
        model,
        condition,
        schedule,
        direction=1,
        sampling_steps=4,
        initial_noise=noise,
        clean_clip=None,
    )
    backward = conditional_latent_ddim_sample(
        model, condition, schedule, direction=-1, sampling_steps=4, initial_noise=noise
    )
    assert forward.shape == backward.shape == condition.shape
    with pytest.raises(ValueError, match="direction"):
        conditional_latent_ddim_sample(model, condition, schedule, direction=0)


def test_conditional_sampler_supports_classifier_free_guidance() -> None:
    model = tiny_model().eval()
    condition = torch.randn(2, 2, 8, 8)
    result = conditional_latent_ddim_sample(
        model,
        condition,
        cosine_alpha_bar(8),
        direction=1,
        sampling_steps=4,
        guidance_scale=1.5,
    )
    assert result.shape == condition.shape


def test_conditional_sampler_can_separate_initial_state_from_condition() -> None:
    model = tiny_model().eval()
    condition = torch.zeros(2, 2, 8, 8)
    result = conditional_latent_ddim_sample(
        model,
        condition,
        cosine_alpha_bar(8),
        direction=-1,
        sampling_steps=4,
        start_fraction=0.75,
        initial_clean=torch.randn_like(condition),
    )
    assert result.shape == condition.shape


def test_denoiser_and_sampler_support_spatial_descriptors() -> None:
    model = BidirectionalLatentDenoiser(
        2,
        base_channels=8,
        time_dim=16,
        latent_size=8,
        descriptor_dim=12,
    ).eval()
    condition = torch.randn(2, 2, 8, 8)
    descriptor = torch.randn(2, 12)
    prediction = model(
        torch.randn_like(condition),
        condition,
        torch.rand(2),
        torch.ones(2),
        descriptor=descriptor,
    )
    assert prediction.shape == condition.shape
    sampled = conditional_latent_ddim_sample(
        model,
        condition,
        cosine_alpha_bar(8),
        direction=1,
        sampling_steps=4,
        descriptor=descriptor,
        guidance_scale=1.5,
    )
    assert sampled.shape == condition.shape
    with pytest.raises(ValueError, match="descriptor must have shape"):
        model(
            torch.randn_like(condition),
            condition,
            torch.rand(2),
            torch.ones(2),
            descriptor=torch.randn(2, 7),
        )
