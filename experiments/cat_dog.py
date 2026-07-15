"""End-to-end CIFAR-10 cat-to-dog feature transport experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brown_bridge.config import ExperimentConfig, FlowConfig, OTConfig
from brown_bridge.data import (
    CAT_LABEL,
    DOG_LABEL,
    BatchPreprocessor,
    BinaryCatDogDataset,
    ClassSubset,
    load_cifar10,
)
from brown_bridge.features import collect_features
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.ot import sinkhorn_divergence
from brown_bridge.paths import (
    brownian_bridge,
    interpolate,
    make_time_grid,
    sample_coupling,
)
from brown_bridge.training import train_classifier


def device_for_host() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(config: ExperimentConfig, checkpoint: Path | None = None) -> dict[str, object]:
    torch.manual_seed(config.seed)
    device = device_for_host()
    raw_train = load_cifar10(config.data_root, train=True)
    raw_test = load_cifar10(config.data_root, train=False)
    train_data = BinaryCatDogDataset(raw_train)
    test_data = BinaryCatDogDataset(raw_test)
    loader_args = dict(batch_size=config.batch_size, num_workers=0)
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        **loader_args,
    )
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)
    preprocess = BatchPreprocessor(config.image_size)
    model = DinoCatDogClassifier(
        config.model_name,
        image_size=config.image_size,
        pretrained=config.pretrained,
        fusion="gated",
    )

    history = None
    if checkpoint is None:
        history = train_classifier(
            model,
            train_loader,
            test_loader,
            preprocess,
            device=device,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    cat_loader = DataLoader(
        ClassSubset(raw_test, CAT_LABEL, config.samples_per_class),
        shuffle=False,
        **loader_args,
    )
    dog_loader = DataLoader(
        ClassSubset(raw_test, DOG_LABEL, config.samples_per_class),
        shuffle=False,
        **loader_args,
    )
    cat_features, cat_indices = collect_features(
        model, cat_loader, preprocess, device=device
    )
    dog_features, dog_indices = collect_features(
        model, dog_loader, preprocess, device=device
    )
    divergence, result = sinkhorn_divergence(
        cat_features,
        dog_features,
        epsilon=config.ot.epsilon,
        max_iterations=config.ot.max_iterations,
        tolerance=config.ot.tolerance,
        cost=config.ot.cost,
    )
    generator = torch.Generator().manual_seed(config.seed)
    endpoints = sample_coupling(
        result.plan,
        cat_features,
        dog_features,
        config.flow.particles,
        generator=generator,
    )
    times = make_time_grid(
        config.flow.macro_steps, config.flow.microsteps_per_macro
    )
    if config.flow.bridge_diffusivity:
        if config.flow.geometry != "ambient" or config.ot.cost != "sqeuclidean":
            raise ValueError(
                "Brownian bridge requires ambient geometry and sqeuclidean cost"
            )
        expected_diffusivity = config.ot.epsilon / 4.0
        if not math.isclose(
            config.flow.bridge_diffusivity,
            expected_diffusivity,
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "for dX=sqrt(2*tau)dW and sqeuclidean cost, "
                f"bridge_diffusivity must equal epsilon/4={expected_diffusivity}"
            )
        trajectories = brownian_bridge(
            endpoints,
            times,
            diffusivity=config.flow.bridge_diffusivity,
            generator=generator,
        )
    else:
        trajectories = interpolate(
            endpoints, times, geometry=config.flow.geometry
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.output_dir / "classifier.pt")
    torch.save(
        {
            "cat_features": cat_features,
            "dog_features": dog_features,
            "cat_indices": cat_indices,
            "dog_indices": dog_indices,
            "plan": result.plan,
            "times": times,
            "source_indices": endpoints.source_indices,
            "target_indices": endpoints.target_indices,
            "trajectories": trajectories,
        },
        config.output_dir / "transport.pt",
    )
    summary: dict[str, object] = {
        "device": str(device),
        "sinkhorn_divergence": float(divergence),
        "transport_cost": float(result.transport_cost),
        "sinkhorn_iterations": result.iterations,
        "sinkhorn_converged": result.converged,
        "max_marginal_error": max(
            float(result.row_marginal_error), float(result.column_marginal_error)
        ),
        "time_points": len(times),
        "particles": config.flow.particles,
        "geometry": config.flow.geometry,
        "best_validation_accuracy": (
            max(history.validation_accuracy) if history is not None else None
        ),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cat_dog"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-class", type=int, default=256)
    parser.add_argument("--macro-steps", type=int, default=10)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument(
        "--geometry", choices=("spherical", "ambient"), default="spherical"
    )
    parser.add_argument(
        "--cost",
        choices=("angular_squared", "cosine", "sqeuclidean"),
        default="angular_squared",
    )
    parser.add_argument("--bridge-diffusivity", type=float, default=0.0)
    parser.add_argument("--random-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        samples_per_class=args.samples_per_class,
        pretrained=not args.random_init,
        ot=OTConfig(epsilon=args.epsilon, cost=args.cost),
        flow=FlowConfig(
            macro_steps=args.macro_steps,
            microsteps_per_macro=args.microsteps,
            particles=args.particles,
            geometry=args.geometry,
            bridge_diffusivity=args.bridge_diffusivity,
        ),
    )
    print(json.dumps(run(config, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
