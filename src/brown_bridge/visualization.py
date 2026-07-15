"""Projection helpers; transport mathematics does not depend on plotting."""

from __future__ import annotations

import torch
from torch import Tensor


def fit_pca_2d(*feature_sets: Tensor) -> tuple[Tensor, Tensor]:
    combined = torch.cat([value.detach().float().cpu() for value in feature_sets])
    center = combined.mean(0, keepdim=True)
    _, _, right_vectors = torch.linalg.svd(combined - center, full_matrices=False)
    return center, right_vectors[:2].T


def project_2d(features: Tensor, center: Tensor, components: Tensor) -> Tensor:
    shape = features.shape
    flat = features.detach().float().cpu().reshape(-1, shape[-1])
    return ((flat - center) @ components).reshape(*shape[:-1], 2)
