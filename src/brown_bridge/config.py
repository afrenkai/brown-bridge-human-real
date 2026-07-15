"""Typed configuration shared by experiments and library code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OTConfig:
    """Numerical and geometric choices for the endpoint coupling."""

    epsilon: float = 0.05
    max_iterations: int = 1_000
    tolerance: float = 1e-7
    cost: str = "angular_squared"

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.cost not in {"cosine", "sqeuclidean", "angular_squared"}:
            raise ValueError(f"unsupported transport cost: {self.cost}")


@dataclass(frozen=True)
class FlowConfig:
    """Time discretization and optional bridge noise."""

    macro_steps: int = 10
    microsteps_per_macro: int = 4
    particles: int = 1_024
    geometry: str = "spherical"
    bridge_diffusivity: float = 0.0

    def __post_init__(self) -> None:
        if self.macro_steps < 1:
            raise ValueError("macro_steps must be positive")
        if self.microsteps_per_macro < 1:
            raise ValueError("microsteps_per_macro must be positive")
        if self.particles < 1:
            raise ValueError("particles must be positive")
        if self.geometry not in {"ambient", "spherical"}:
            raise ValueError(f"unsupported geometry: {self.geometry}")
        if self.bridge_diffusivity < 0:
            raise ValueError("bridge_diffusivity cannot be negative")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the CIFAR-10 cat/dog demonstration."""

    data_root: Path
    output_dir: Path = Path("outputs/cat_dog")
    samples_per_class: int = 256
    image_size: int = 64
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    seed: int = 0
    model_name: str = "vit_tiny_patch16_dinov3_qkvb"
    pretrained: bool = True
    ot: OTConfig = field(default_factory=OTConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)

    def __post_init__(self) -> None:
        if self.samples_per_class < 1:
            raise ValueError("samples_per_class must be positive")
        if self.image_size < 1 or self.image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
