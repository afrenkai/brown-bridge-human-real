"""Held-out reciprocal cat↔dog bridge with content-aware DINO coupling."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from brown_bridge.data import CAT_LABEL, DOG_LABEL, BatchPreprocessor, ClassSubset, load_cifar10
from brown_bridge.features import (
    collect_spatial_features,
    content_cost_matrix,
    content_descriptor,
    project_off_direction,
    species_direction,
)
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.ot import pairwise_cost, sinkhorn, sinkhorn_divergence, sinkhorn_from_cost
from brown_bridge.reciprocal import (
    DirectionalTimeMLP,
    integrate_directional_sde,
    make_reciprocal_batch,
    orthogonal_displacement,
    shared_noise_cycle_loss,
    species_progression_loss,
)


SEED = 73
FEATURE_DIM = 192
FIT_COUNT = 600
VALIDATION_COUNT = 200
TEST_COUNT = 200
TAU = 0.0125
EPSILON = 0.10
BATCH_SIZE = 256
TERMINAL_BATCH = 48
TERMINAL_STEPS = 6
TERMINAL_EVERY = 8
SPECIES_WEIGHT = 0.05
TERMINAL_WEIGHT = 50.0
LAMBDA_CONTENT = 2.0
LAMBDA_RECIPROCAL = 2.0
TRAIN_KAPPA = 0.0025
EVAL_STEPS = 80
OUTPUT = Path("outputs/reciprocal_cat_dog")
CLASSIFIER = Path("outputs/cat_dog/classifier.pt")


@dataclass(frozen=True)
class Split:
    cat_global: Tensor
    dog_global: Tensor
    cat_patch: Tensor
    dog_patch: Tensor
    cat_indices: Tensor
    dog_indices: Tensor


def host_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_features(data_root: Path, output: Path, device: torch.device) -> dict[str, Tensor]:
    cache = output / "dino_spatial_features.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)
    raw = load_cifar10(data_root, train=False)
    model = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb",
        image_size=64,
        pretrained=False,
        fusion="gated",
    )
    model.load_state_dict(torch.load(CLASSIFIER, map_location="cpu", weights_only=True))
    preprocess = BatchPreprocessor(64)
    loader_args = {"batch_size": 128, "shuffle": False, "num_workers": 0}
    cat_global, cat_patch, cat_indices = collect_spatial_features(
        model,
        DataLoader(ClassSubset(raw, CAT_LABEL, 1_000), **loader_args),
        preprocess,
        device=device,
    )
    dog_global, dog_patch, dog_indices = collect_spatial_features(
        model,
        DataLoader(ClassSubset(raw, DOG_LABEL, 1_000), **loader_args),
        preprocess,
        device=device,
    )
    result = {
        "cat_global": cat_global,
        "dog_global": dog_global,
        "cat_patch": cat_patch,
        "dog_patch": dog_patch,
        "cat_indices": cat_indices,
        "dog_indices": dog_indices,
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache)
    return result


def make_splits(features: dict[str, Tensor]) -> dict[str, Split]:
    generator = torch.Generator().manual_seed(SEED)
    cat_order = torch.randperm(1_000, generator=generator)
    dog_order = torch.randperm(1_000, generator=generator)
    boundaries = (0, FIT_COUNT, FIT_COUNT + VALIDATION_COUNT, 1_000)
    result: dict[str, Split] = {}
    for name, left, right in zip(("fit", "validation", "test"), boundaries[:-1], boundaries[1:]):
        cat = cat_order[left:right]
        dog = dog_order[left:right]
        result[name] = Split(
            features["cat_global"][cat],
            features["dog_global"][dog],
            features["cat_patch"][cat],
            features["dog_patch"][dog],
            features["cat_indices"][cat],
            features["dog_indices"][dog],
        )
    return result


def content_plan(split: Split, direction: Tensor) -> tuple[Tensor, Tensor, dict[str, float]]:
    cats = content_descriptor(split.cat_global, split.cat_patch, direction)
    dogs = content_descriptor(split.dog_global, split.dog_patch, direction)
    cost = content_cost_matrix(cats, dogs)
    result = sinkhorn_from_cost(
        cost,
        epsilon=EPSILON,
        max_iterations=2_000,
        tolerance=1e-8,
    )
    global_result = sinkhorn(
        split.cat_global,
        split.dog_global,
        epsilon=0.05,
        cost="sqeuclidean",
        max_iterations=2_000,
        tolerance=1e-8,
    )
    product = torch.full_like(result.plan, 1.0 / result.plan.numel())
    diagnostics = {
        "content_ot_expected_cost": float((result.plan * cost).sum()),
        "global_ot_expected_content_cost": float((global_result.plan * cost).sum()),
        "random_expected_content_cost": float((product * cost).sum()),
        "content_ot_vs_global_reduction": 1.0
        - float((result.plan * cost).sum() / (global_result.plan * cost).sum()),
        "content_ot_vs_random_reduction": 1.0
        - float((result.plan * cost).sum() / (product * cost).sum()),
    }
    return result.plan, cost, diagnostics


def sliced_wasserstein(source: Tensor, target: Tensor, directions: Tensor) -> Tensor:
    source_projection = torch.sort(source @ directions.T, dim=0).values
    target_projection = torch.sort(target @ directions.T, dim=0).values
    return (source_projection - target_projection).square().mean()


def fit_probe(split: Split) -> nn.Linear:
    """Independent species probe fitted only on the validation partition."""
    features = torch.cat([split.cat_global, split.dog_global])
    labels = torch.cat(
        [torch.zeros(len(split.cat_global)), torch.ones(len(split.dog_global))]
    )
    probe = nn.Linear(FEATURE_DIM, 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(500):
        logits = probe(features).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


def train_variant(
    name: str,
    split: Split,
    plan: Tensor,
    species: Tensor,
    *,
    steps: int,
    use_regularizers: bool,
    device: torch.device,
) -> tuple[DirectionalTimeMLP, DirectionalTimeMLP, list[dict[str, float]]]:
    torch.manual_seed(SEED + (1 if use_regularizers else 0))
    velocity = DirectionalTimeMLP(FEATURE_DIM, width=256, depth=3, time_frequencies=3).to(device)
    score = DirectionalTimeMLP(FEATURE_DIM, width=256, depth=3, time_frequencies=3).to(device)
    optimizer = torch.optim.AdamW(
        [*velocity.parameters(), *score.parameters()], lr=3e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(SEED + 11 + int(use_regularizers))
    sw_directions = F.normalize(
        torch.randn(48, FEATURE_DIM, generator=generator), dim=1
    ).to(device)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        batch = make_reciprocal_batch(
            plan,
            split.cat_global,
            split.dog_global,
            BATCH_SIZE,
            diffusivity=TAU,
            generator=generator,
            device=device,
        )
        predicted_velocity = velocity(batch.state, batch.time, batch.direction)
        predicted_score = score(batch.state, batch.time, batch.direction)
        flow_per_sample = (predicted_velocity - batch.conditional_velocity).square().mean(1)
        flow_loss = (batch.flow_weight * flow_per_sample).sum() / batch.flow_weight.sum()
        gamma = torch.sqrt(2.0 * TAU * batch.time * (1.0 - batch.time)).clamp_min(1e-6)
        score_loss = F.mse_loss(gamma[:, None] * predicted_score, batch.scaled_score)
        species_loss = species_progression_loss(
            predicted_velocity, batch.direction, species.to(device)
        )
        terminal_loss = torch.zeros((), device=device)
        content_loss = torch.zeros((), device=device)
        reciprocal_loss = torch.zeros((), device=device)
        if step % TERMINAL_EVERY == 0:
            cat_choice = torch.randint(FIT_COUNT, (TERMINAL_BATCH,), generator=generator)
            dog_choice = torch.randint(FIT_COUNT, (TERMINAL_BATCH,), generator=generator)
            cats = split.cat_global[cat_choice].to(device)
            dogs = split.dog_global[dog_choice].to(device)
            forward = integrate_directional_sde(
                velocity, score, cats, direction=1, steps=TERMINAL_STEPS, kappa=0.0
            )
            backward = integrate_directional_sde(
                velocity, score, dogs, direction=-1, steps=TERMINAL_STEPS, kappa=0.0
            )
            terminal_loss = 0.5 * (
                sliced_wasserstein(forward[-1], dogs, sw_directions)
                + sliced_wasserstein(backward[-1], cats, sw_directions)
            )
            if use_regularizers:
                reciprocal_loss, noisy_forward, _ = shared_noise_cycle_loss(
                    velocity,
                    score,
                    cats,
                    steps=TERMINAL_STEPS,
                    kappa=TRAIN_KAPPA,
                    generator=generator,
                )
                content_loss = orthogonal_displacement(
                    noisy_forward[-1], cats, species.to(device)
                )
        total = (
            flow_loss
            + score_loss
            + TERMINAL_WEIGHT * terminal_loss
            + SPECIES_WEIGHT * species_loss
        )
        if use_regularizers:
            total = (
                total
                + LAMBDA_CONTENT * TERMINAL_EVERY * content_loss
                + LAMBDA_RECIPROCAL * TERMINAL_EVERY * reciprocal_loss
            )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([*velocity.parameters(), *score.parameters()], 1.0)
        optimizer.step()
        scheduler.step()
        history.append(
            {
                "total": float(total.detach()),
                "flow": float(flow_loss.detach()),
                "score": float(score_loss.detach()),
                "species": float(species_loss.detach()),
                "terminal": float(terminal_loss.detach()),
                "content": float(content_loss.detach()),
                "reciprocal": float(reciprocal_loss.detach()),
            }
        )
        if step == 1 or step % 250 == 0:
            recent = history[-100:]
            means = {key: sum(item[key] for item in recent) / len(recent) for key in recent[0]}
            print(
                f"{name} step={step:04d} flow={means['flow']:.5f} "
                f"terminal={means['terminal']:.5f} content={means['content']:.5f} "
                f"reciprocal={means['reciprocal']:.5f}",
                flush=True,
            )
    return velocity.eval(), score.eval(), history


def sinkhorn_value(source: Tensor, target: Tensor) -> float:
    value, _ = sinkhorn_divergence(
        source,
        target,
        epsilon=0.05,
        cost="sqeuclidean",
        max_iterations=1_000,
        tolerance=1e-7,
    )
    return float(value)


@torch.inference_mode()
def evaluate_variant(
    velocity: DirectionalTimeMLP,
    score: DirectionalTimeMLP,
    split: Split,
    species: Tensor,
    probe: nn.Linear,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    cats = split.cat_global.to(device)
    dogs = split.dog_global.to(device)
    forward = integrate_directional_sde(
        velocity, score, cats, direction=1, steps=EVAL_STEPS, kappa=0.0
    ).cpu()
    backward = integrate_directional_sde(
        velocity, score, dogs, direction=-1, steps=EVAL_STEPS, kappa=0.0
    ).cpu()
    cat_cycle = integrate_directional_sde(
        velocity,
        score,
        forward[-1].to(device),
        direction=-1,
        steps=EVAL_STEPS,
        kappa=0.0,
    ).cpu()
    dog_cycle = integrate_directional_sde(
        velocity,
        score,
        backward[-1].to(device),
        direction=1,
        steps=EVAL_STEPS,
        kappa=0.0,
    ).cpu()
    sample_indices = list(range(0, EVAL_STEPS + 1, EVAL_STEPS // 10))
    times = [index / EVAL_STEPS for index in sample_indices]
    forward_sinkhorn = [sinkhorn_value(forward[index], split.dog_global) for index in sample_indices]
    backward_sinkhorn = [sinkhorn_value(backward[index], split.cat_global) for index in sample_indices]
    forward_logits = probe(forward.reshape(-1, FEATURE_DIM)).reshape(len(forward), -1)
    backward_logits = probe(backward.reshape(-1, FEATURE_DIM)).reshape(len(backward), -1)
    forward_monotone = float((forward_logits[1:] >= forward_logits[:-1]).float().mean())
    backward_monotone = float((backward_logits[1:] <= backward_logits[:-1]).float().mean())
    forward_probability = forward_logits.sigmoid().mean(1)
    backward_probability = backward_logits.sigmoid().mean(1)
    target_dog_probability = float(probe(split.dog_global).sigmoid().mean())
    target_cat_probability = float(probe(split.cat_global).sigmoid().mean())
    unit = F.normalize(species, dim=0)
    forward_content = project_off_direction(forward - forward[0:1], unit).square().mean((1, 2))
    backward_content = project_off_direction(backward - backward[0:1], unit).square().mean((1, 2))

    stochastic_cycles: list[float] = []
    stochastic_endpoints: list[Tensor] = []
    subset = cats[:32]
    for seed in range(8):
        generator = torch.Generator().manual_seed(900 + seed)
        noise = torch.randn(EVAL_STEPS, len(subset), FEATURE_DIM, generator=generator).to(device)
        stochastic_forward = integrate_directional_sde(
            velocity,
            score,
            subset,
            direction=1,
            steps=EVAL_STEPS,
            kappa=TRAIN_KAPPA,
            noise=noise,
        )
        stochastic_reverse = integrate_directional_sde(
            velocity,
            score,
            stochastic_forward[-1],
            direction=-1,
            steps=EVAL_STEPS,
            kappa=TRAIN_KAPPA,
            noise=-noise.flip(0),
        )
        stochastic_cycles.append(float(F.mse_loss(stochastic_reverse[-1], subset)))
        stochastic_endpoints.append(stochastic_forward[-1].cpu())
    endpoint_stack = torch.stack(stochastic_endpoints)
    diversity = float(endpoint_stack.var(0, correction=0).mean())

    test_features = torch.cat([split.cat_global, split.dog_global])
    test_labels = torch.cat([torch.zeros(TEST_COUNT), torch.ones(TEST_COUNT)])
    probe_accuracy = float(
        ((probe(test_features).flatten() > 0).float() == test_labels).float().mean()
    )
    metrics: dict[str, object] = {
        "independent_probe_test_accuracy": probe_accuracy,
        "forward": {
            "sinkhorn_before": forward_sinkhorn[0],
            "sinkhorn_terminal": forward_sinkhorn[-1],
            "sinkhorn_relative_reduction": 1.0 - forward_sinkhorn[-1] / forward_sinkhorn[0],
            "terminal_dog_probability": float(forward_probability[-1]),
            "target_dog_probability": target_dog_probability,
            "probability_calibration_error": abs(
                float(forward_probability[-1]) - target_dog_probability
            ),
            "monotone_step_fraction": forward_monotone,
            "terminal_content_mse": float(forward_content[-1]),
            "deterministic_cycle_mse": float(F.mse_loss(cat_cycle[-1], split.cat_global)),
        },
        "backward": {
            "sinkhorn_before": backward_sinkhorn[0],
            "sinkhorn_terminal": backward_sinkhorn[-1],
            "sinkhorn_relative_reduction": 1.0 - backward_sinkhorn[-1] / backward_sinkhorn[0],
            "terminal_dog_probability": float(backward_probability[-1]),
            "target_dog_probability": target_cat_probability,
            "probability_calibration_error": abs(
                float(backward_probability[-1]) - target_cat_probability
            ),
            "monotone_step_fraction": backward_monotone,
            "terminal_content_mse": float(backward_content[-1]),
            "deterministic_cycle_mse": float(F.mse_loss(dog_cycle[-1], split.dog_global)),
        },
        "shared_noise_cycle_mse_mean": sum(stochastic_cycles) / len(stochastic_cycles),
        "shared_noise_cycle_mse_values": stochastic_cycles,
        "conditional_endpoint_variance": diversity,
        "curves": {
            "time": times,
            "forward_sinkhorn": forward_sinkhorn,
            "backward_sinkhorn": backward_sinkhorn,
            "forward_dog_probability": [float(forward_probability[index]) for index in sample_indices],
            "backward_dog_probability": [float(backward_probability[index]) for index in sample_indices],
            "forward_content_mse": [float(forward_content[index]) for index in sample_indices],
            "backward_content_mse": [float(backward_content[index]) for index in sample_indices],
        },
    }
    return metrics, {"forward": forward, "backward": backward}


def plot_metrics(summary: dict[str, object], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    styles = {"baseline": "--", "regularized": "-"}
    for name in ("baseline", "regularized"):
        curves = summary["variants"][name]["curves"]
        axes[0].plot(curves["time"], curves["forward_sinkhorn"], styles[name], label=f"{name} cat→dog")
        axes[0].plot(curves["time"], curves["backward_sinkhorn"], styles[name], alpha=0.65, label=f"{name} dog→cat")
        axes[1].plot(curves["time"], curves["forward_dog_probability"], styles[name], label=f"{name} cat→dog")
        axes[1].plot(curves["time"], curves["backward_dog_probability"], styles[name], alpha=0.65, label=f"{name} dog→cat")
        axes[2].plot(curves["time"], curves["forward_content_mse"], styles[name], label=f"{name} cat→dog")
        axes[2].plot(curves["time"], curves["backward_content_mse"], styles[name], alpha=0.65, label=f"{name} dog→cat")
    axes[0].set(xlabel="time", ylabel="Sinkhorn divergence", title="Held-out target alignment")
    axes[1].set(xlabel="time", ylabel="independent P(dog)", title="Species progression", ylim=(-0.03, 1.03))
    axes[2].set(xlabel="time", ylabel="orthogonal feature MSE", title="Content displacement")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    figure.savefig(output / "reciprocal_diagnostics.png", dpi=180)
    plt.close(figure)


def plot_pairings(
    data_root: Path,
    split: Split,
    plan: Tensor,
    global_plan: Tensor,
    output: Path,
) -> None:
    raw = load_cifar10(data_root, train=False)
    rows = 6
    sources = torch.linspace(0, len(split.cat_indices) - 1, rows).long()
    content_targets = plan[sources].argmax(1)
    global_targets = global_plan[sources].argmax(1)
    figure, axes = plt.subplots(rows, 3, figsize=(4.8, 1.6 * rows), constrained_layout=True)
    for row, (source, content_target, global_target) in enumerate(
        zip(sources, content_targets, global_targets)
    ):
        indices = (
            int(split.cat_indices[source]),
            int(split.dog_indices[content_target]),
            int(split.dog_indices[global_target]),
        )
        for column, index in enumerate(indices):
            image, _ = raw[index]
            axes[row, column].imshow(to_tensor(image).permute(1, 2, 0))
            axes[row, column].axis("off")
    for column, title in enumerate(("source cat", "content-OT dog", "global-OT dog")):
        axes[0, column].set_title(title, fontsize=9)
    figure.savefig(output / "pairing_comparison.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER)
    parser.add_argument("--steps", type=int, default=1_500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUTPUT, CLASSIFIER
    OUTPUT = args.output_dir.resolve()
    CLASSIFIER = args.classifier.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = host_device()
    features = extract_features(args.data_root, OUTPUT, device)
    splits = make_splits(features)
    species = species_direction(splits["fit"].cat_global, splits["fit"].dog_global)
    plan, content_cost, coupling_metrics = content_plan(splits["fit"], species)
    global_result = sinkhorn(
        splits["fit"].cat_global,
        splits["fit"].dog_global,
        epsilon=0.05,
        cost="sqeuclidean",
        max_iterations=2_000,
        tolerance=1e-8,
    )
    probe = fit_probe(splits["validation"])
    variants: dict[str, object] = {}
    checkpoints: dict[str, object] = {}
    trajectories: dict[str, object] = {}
    for name, regularized in (("baseline", False), ("regularized", True)):
        velocity, score, history = train_variant(
            name,
            splits["fit"],
            plan,
            species,
            steps=args.steps,
            use_regularizers=regularized,
            device=device,
        )
        metrics, paths = evaluate_variant(
            velocity,
            score,
            splits["test"],
            species,
            probe,
            device=device,
        )
        metrics["recent_100_losses"] = {
            key: sum(item[key] for item in history[-100:]) / min(100, len(history))
            for key in history[-1]
        }
        variants[name] = metrics
        checkpoints[name] = {
            "velocity": velocity.state_dict(),
            "score": score.state_dict(),
            "history": history,
        }
        trajectories[name] = paths
    summary: dict[str, object] = {
        "device": str(device),
        "dataset": "CIFAR-10 held-out test cats and dogs",
        "split_per_class": {"fit": FIT_COUNT, "validation": VALIDATION_COUNT, "test": TEST_COUNT},
        "training_steps_per_variant": args.steps,
        "bridge": {
            "reference_diffusivity": TAU,
            "content_ot_epsilon": EPSILON,
            "shared_directional_velocity_and_score": True,
            "species_weight": SPECIES_WEIGHT,
            "terminal_distribution_weight": TERMINAL_WEIGHT,
            "lambda_content": LAMBDA_CONTENT,
            "lambda_reciprocal": LAMBDA_RECIPROCAL,
            "regularizer_kappa": TRAIN_KAPPA,
        },
        "coupling": coupling_metrics,
        "variants": variants,
    }
    torch.save(
        {
            "checkpoints": checkpoints,
            "trajectories": trajectories,
            "species_direction": species,
            "fit_plan": plan,
            "fit_content_cost": content_cost,
            "test_split": splits["test"],
            "probe": probe.state_dict(),
        },
        OUTPUT / "reciprocal_bridge.pt",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_metrics(summary, OUTPUT)
    plot_pairings(args.data_root, splits["fit"], plan, global_result.plan, OUTPUT)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
