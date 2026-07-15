"""CIFAR-10 data adapters; no experiment state is created at import time."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CAT_LABEL = 3
DOG_LABEL = 5
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def resolve_data_root(start: Path) -> Path:
    """Find an extracted ``cifar-10-batches-py`` in an ancestor data dir."""
    for parent in (start.resolve(), *start.resolve().parents):
        candidate = parent / "data"
        if (candidate / "cifar-10-batches-py").is_dir():
            return candidate
    raise FileNotFoundError("could not find data/cifar-10-batches-py")


def load_cifar10(root: Path, *, train: bool) -> datasets.CIFAR10:
    return datasets.CIFAR10(root=root, train=train, download=False, transform=None)


class BinaryCatDogDataset(Dataset[tuple[Tensor, int, int]]):
    """Raw tensor images with binary labels and original dataset indices."""

    def __init__(self, dataset: datasets.CIFAR10) -> None:
        self.dataset = dataset
        self.to_tensor: Callable = transforms.ToTensor()
        self.indices = [
            index
            for index, label in enumerate(dataset.targets)
            if label in (CAT_LABEL, DOG_LABEL)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int, int]:
        index = self.indices[item]
        image, label = self.dataset[index]
        return self.to_tensor(image), int(label == DOG_LABEL), index


class ClassSubset(Dataset[tuple[Tensor, int]]):
    """Deterministic class subset returning each original dataset index."""

    def __init__(
        self,
        dataset: datasets.CIFAR10,
        class_id: int,
        max_samples: int,
    ) -> None:
        self.dataset = dataset
        self.to_tensor = transforms.ToTensor()
        self.indices = [
            index for index, label in enumerate(dataset.targets) if label == class_id
        ][:max_samples]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int]:
        index = self.indices[item]
        image, _ = self.dataset[index]
        return self.to_tensor(image), index


class BatchPreprocessor:
    """Resize, augment, and normalize a whole tensor batch on its device."""

    def __init__(
        self,
        image_size: int,
        mean: tuple[float, float, float] = CIFAR10_MEAN,
        std: tuple[float, float, float] = CIFAR10_STD,
    ) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, images: Tensor, *, training: bool) -> Tensor:
        images = images.clone()
        if training:
            flip = torch.rand(len(images), device=images.device) < 0.5
            images[flip] = images[flip].flip(3)
            brightness = 1.0 + 0.2 * (
                2.0 * torch.rand(len(images), 1, 1, 1, device=images.device) - 1.0
            )
            contrast = 1.0 + 0.2 * (
                2.0 * torch.rand(len(images), 1, 1, 1, device=images.device) - 1.0
            )
            channel_mean = images.mean((2, 3), keepdim=True)
            images = ((images - channel_mean) * contrast + channel_mean) * brightness
            images = images.clamp(0.0, 1.0)
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (images - self.mean.to(images)) / self.std.to(images)
