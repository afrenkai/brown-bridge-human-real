import torch

from brown_bridge.latent_image_bridge import (
    LatentImageAutoencoder,
    SpatialReciprocalField,
    integrate_spatial_probability_flow,
    make_spatial_reciprocal_batch,
)


def test_latent_autoencoder_shapes() -> None:
    model = LatentImageAutoencoder(latent_channels=4, base_channels=8)
    images = torch.rand(2, 3, 128, 128)
    reconstruction, latent = model(images)
    assert latent.shape == (2, 4, 32, 32)
    assert reconstruction.shape == images.shape
    assert 0 <= reconstruction.min() <= reconstruction.max() <= 1


def test_spatial_reciprocal_batch_is_endpoint_symmetric() -> None:
    source = torch.zeros(6, 2, 8, 8)
    target = torch.ones_like(source)
    batch = make_spatial_reciprocal_batch(
        source, target, diffusivity=0.01, generator=torch.Generator().manual_seed(4)
    )
    assert batch.state.shape == source.shape
    assert set(batch.direction.tolist()) <= {-1, 1}
    assert torch.isfinite(batch.conditional_velocity).all()
    assert torch.all(batch.flow_weight > 0)


def test_zero_spatial_field_has_constant_probability_flow() -> None:
    field = SpatialReciprocalField(
        latent_channels=2, width=8, depth=1, condition_dim=8
    )
    for parameter in field.parameters():
        torch.nn.init.zeros_(parameter)
    initial = torch.randn(2, 2, 8, 8)
    path = integrate_spatial_probability_flow(field, initial, steps=3)
    assert torch.allclose(path, initial.expand_as(path))
