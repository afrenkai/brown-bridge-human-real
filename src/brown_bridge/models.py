"""DINO-backed classifiers and feature adapters."""

from __future__ import annotations

from typing import Literal

import timm
import torch
from torch import Tensor, nn


def cls_features(model: nn.Module, images: Tensor) -> Tensor:
    """Extract a pre-classifier CLS representation from common DINO outputs."""
    if not hasattr(model, "forward_features"):
        raise TypeError("model must expose forward_features")
    output = model.forward_features(images)
    if isinstance(output, Tensor):
        features = output[:, 0] if output.ndim == 3 else output
    elif isinstance(output, dict):
        keys = ("x_norm_clstoken", "cls_token", "cls_features", "features")
        key = next((candidate for candidate in keys if candidate in output), None)
        if key is None:
            raise KeyError(f"could not find CLS features in {tuple(output)}")
        features = output[key]
    else:
        raise TypeError(f"unsupported feature output: {type(output).__name__}")
    if features.ndim != 2 or features.shape[1] <= 2:
        raise ValueError(f"expected [batch, feature_dim], received {features.shape}")
    return features


class DinoCatDogClassifier(nn.Module):
    """DINO encoder with either CLS-only or learned CLS/patch fusion."""

    def __init__(
        self,
        model_name: str,
        *,
        image_size: int,
        pretrained: bool = True,
        fusion: Literal["cls", "gated"] = "gated",
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
        dimension = int(self.backbone.num_features)
        self.cls_norm = nn.LayerNorm(dimension)
        if fusion == "gated":
            self.patch_norm = nn.LayerNorm(dimension)
            self.gate = nn.Sequential(nn.Linear(2 * dimension, dimension), nn.Sigmoid())
            self.fusion_norm = nn.LayerNorm(dimension)
        elif fusion != "cls":
            raise ValueError(f"unsupported fusion: {fusion}")
        self.classifier = nn.Sequential(nn.Dropout(0.15), nn.Linear(dimension, 2))

    def encode_spatial(
        self, images: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Return fused image features, gates, and normalized patch tokens."""
        tokens = self.backbone.forward_features(images)
        if isinstance(tokens, dict):
            key = next(
                (
                    candidate
                    for candidate in (
                        "x_norm_clstoken",
                        "cls_token",
                        "cls_features",
                        "features",
                    )
                    if candidate in tokens
                ),
                None,
            )
            if key is None:
                raise KeyError(f"could not find CLS features in {tuple(tokens)}")
            features = self.cls_norm(tokens[key])
            return features, None, None
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise ValueError("DINO backbone must return token tensor or feature dict")
        cls = self.cls_norm(tokens[:, 0])
        if self.fusion == "cls":
            prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
            return cls, None, tokens[:, prefix:]
        prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        raw_patches = tokens[:, prefix:]
        patch_summary = self.patch_norm(raw_patches.mean(1))
        patches = self.patch_norm(raw_patches)
        gate = self.gate(torch.cat([cls, patch_summary], dim=1))
        features = self.fusion_norm(gate * cls + (1.0 - gate) * patch_summary)
        return features, gate, patches

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor | None]:
        features, gate, _ = self.encode_spatial(images)
        return features, gate

    def forward_spatial_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features, _, patches = self.encode_spatial(images)
        if patches is None:
            raise ValueError("the selected backbone does not expose patch tokens")
        return features, patches

    def forward_features(self, images: Tensor) -> Tensor:
        return self.encode(images)[0]

    def forward(
        self,
        images: Tensor,
        *,
        return_gate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        features, gate = self.encode(images)
        logits = self.classifier(features)
        return (logits, gate) if return_gate else logits
