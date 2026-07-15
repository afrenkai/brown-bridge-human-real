import torch

from brown_bridge.residual_diffusion import (
    ResidualDenoiser,
    cosine_alpha_bar,
    ddim_sample,
    q_sample,
    refine_image,
)


def test_cosine_schedule_is_decreasing_and_bounded() -> None:
    schedule = cosine_alpha_bar(20)
    assert schedule.shape == (20,)
    assert torch.all(schedule[1:] < schedule[:-1])
    assert 0 < schedule[-1] < schedule[0] < 1


def test_q_sample_has_expected_shape() -> None:
    clean = torch.ones(2, 3, 16, 16)
    noise = torch.zeros_like(clean)
    schedule = cosine_alpha_bar(10)
    sampled = q_sample(clean, noise, torch.tensor([0, 9]), schedule)
    assert sampled.shape == clean.shape
    assert sampled[0].mean() > sampled[1].mean()


def test_denoiser_and_short_ddim_are_finite() -> None:
    model = ResidualDenoiser(base_channels=8, time_dim=32).eval()
    condition = torch.rand(2, 3, 16, 16).mul(2).sub(1)
    noise = torch.randn_like(condition)
    prediction = model(noise, condition, torch.tensor([0.2, 0.8]))
    assert prediction.shape == condition.shape
    sampled = ddim_sample(
        model,
        condition,
        cosine_alpha_bar(10),
        sampling_steps=3,
        initial_noise=noise,
    )
    assert sampled.shape == condition.shape
    assert torch.isfinite(sampled).all()
    truncated = ddim_sample(
        model,
        condition,
        cosine_alpha_bar(10),
        sampling_steps=3,
        initial_noise=noise,
        start_fraction=0.4,
    )
    assert torch.isfinite(truncated).all()


def test_refinement_clamps_to_image_range() -> None:
    base = torch.full((1, 3, 8, 8), 0.5)
    residual = torch.full_like(base, 4.0)
    assert refine_image(base, residual).max() == 1.0
