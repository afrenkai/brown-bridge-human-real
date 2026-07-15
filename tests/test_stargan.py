import torch

from brown_bridge.stargan import Generator, StyleEncoder


def test_small_renderer_shapes() -> None:
    generator = Generator(image_size=32, style_dim=8, max_conv_dim=64)
    encoder = StyleEncoder(
        image_size=32, style_dim=8, num_domains=3, max_conv_dim=64
    )
    image = torch.randn(2, 3, 32, 32)
    domain = torch.tensor([0, 2])
    style = encoder(image, domain)
    rendered = generator(image, style)
    assert style.shape == (2, 8)
    assert rendered.shape == image.shape


def test_repeated_layer_styles_match_global_style() -> None:
    torch.manual_seed(7)
    generator = Generator(image_size=32, style_dim=8, max_conv_dim=64).eval()
    image = torch.randn(2, 3, 32, 32)
    style = torch.randn(2, 8)
    repeated = style[:, None].expand(-1, len(generator.decode), -1)
    assert torch.allclose(
        generator(image, style), generator(image, repeated), atol=1e-6
    )


def test_renderer_rejects_non_power_of_two() -> None:
    try:
        Generator(image_size=48)
    except ValueError as error:
        assert "power of two" in str(error)
    else:
        raise AssertionError("expected invalid image size to fail")
