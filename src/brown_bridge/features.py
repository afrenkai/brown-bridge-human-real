"""Feature extraction independent of a particular dataset or notebook."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    preprocess: Callable[..., Tensor],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect normalized feature rows and the corresponding dataset indices."""
    model.eval().to(device)
    feature_batches: list[Tensor] = []
    index_batches: list[Tensor] = []
    for images, indices in loader:
        images = preprocess(images.to(device, non_blocking=True), training=False)
        features = model.forward_features(images)
        if features.ndim != 2:
            raise ValueError("forward_features must return [batch, feature_dim]")
        feature_batches.append(F.normalize(features.float(), dim=1).cpu())
        index_batches.append(indices.cpu())
    return torch.cat(feature_batches), torch.cat(index_batches)
