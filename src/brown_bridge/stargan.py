"""Minimal StarGAN-v2 AFHQ renderer used as an image-space observation model.

The architecture is adapted from the official StarGAN-v2 implementation by
NAVER (https://github.com/clovaai/stargan-v2), licensed CC BY-NC 4.0.  Only
the generator and reference style encoder are retained; training machinery,
the discriminator, FAN, and DataParallel wrappers are deliberately omitted.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResBlk(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        normalize: bool = False,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, bias=False)

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x: Tensor) -> Tensor:
        if self.normalize:
            x = self.norm1(x)
        x = self.conv1(self.actv(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        return self.conv2(self.actv(x))

    def forward(self, x: Tensor) -> Tensor:
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2.0)


class AdaIN(nn.Module):
    def __init__(self, style_dim: int, num_features: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, 2 * num_features)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        affine = self.fc(style).view(style.shape[0], -1, 1, 1)
        gamma, beta = affine.chunk(2, dim=1)
        return (1.0 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        style_dim: int = 64,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, bias=False)

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: Tensor, style: Tensor) -> Tensor:
        x = self.actv(self.norm1(x, style))
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(self.actv(self.norm2(x, style)))
        return x

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        return (self._residual(x, style) + self._shortcut(x)) / math.sqrt(2.0)


class Generator(nn.Module):
    """StarGAN-v2 generator without facial high-pass masks."""

    def __init__(
        self,
        image_size: int = 256,
        style_dim: int = 64,
        max_conv_dim: int = 512,
    ) -> None:
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 32")
        dim_in = 2**14 // image_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1),
        )
        repeat_num = int(math.log2(image_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(2 * dim_in, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True)
            )
            self.decode.insert(
                0,
                AdainResBlk(
                    dim_out, dim_in, style_dim=style_dim, upsample=True
                ),
            )
            dim_in = dim_out
        for _ in range(2):
            self.encode.append(ResBlk(dim_in, dim_in, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_in, dim_in, style_dim=style_dim)
            )

    def forward(self, image: Tensor, style: Tensor) -> Tensor:
        """Render with one global style or one style per decoder block.

        ``style`` may be ``[batch, style_dim]`` for the original StarGAN-v2
        behavior or ``[batch, decoder_blocks, style_dim]`` for lightweight
        coarse-to-fine bridge conditioning.
        """
        if style.ndim == 2:
            if len(style) != len(image):
                raise ValueError("global style batch must match image batch")
        elif style.ndim == 3:
            if style.shape[:2] != (len(image), len(self.decode)):
                raise ValueError(
                    "layerwise style must have shape "
                    f"[batch, {len(self.decode)}, style_dim]"
                )
        else:
            raise ValueError("style must be rank two or three")
        hidden = self.from_rgb(image)
        for block in self.encode:
            hidden = block(hidden)
        for index, block in enumerate(self.decode):
            block_style = style if style.ndim == 2 else style[:, index]
            hidden = block(hidden, block_style)
        return self.to_rgb(hidden)


class StyleEncoder(nn.Module):
    """Extract a domain-specific StarGAN style vector from a reference image."""

    def __init__(
        self,
        image_size: int = 256,
        style_dim: int = 64,
        num_domains: int = 3,
        max_conv_dim: int = 512,
    ) -> None:
        super().__init__()
        dim_in = 2**14 // image_size
        blocks: list[nn.Module] = [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(math.log2(image_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(2 * dim_in, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        blocks.extend(
            [
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim_in, dim_in, 4),
                nn.LeakyReLU(0.2),
            ]
        )
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.ModuleList(
            nn.Linear(dim_in, style_dim) for _ in range(num_domains)
        )

    def forward(self, image: Tensor, domain: Tensor) -> Tensor:
        hidden = self.shared(image).flatten(1)
        styles = torch.stack([layer(hidden) for layer in self.unshared], dim=1)
        rows = torch.arange(len(domain), device=domain.device)
        return styles[rows, domain]


def load_afhq_renderer(
    checkpoint: str,
    *,
    device: torch.device,
) -> tuple[Generator, StyleEncoder]:
    """Load official AFHQ EMA weights into MPS-safe unwrapped modules."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    generator = Generator().eval()
    style_encoder = StyleEncoder().eval()
    generator.load_state_dict(state["generator"], strict=True)
    style_encoder.load_state_dict(state["style_encoder"], strict=True)
    generator.requires_grad_(False).to(device)
    style_encoder.requires_grad_(False).to(device)
    return generator, style_encoder
