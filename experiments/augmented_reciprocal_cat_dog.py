"""Reciprocal cat↔dog bridge with an augmented endpoint constraint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

import reciprocal_cat_dog as base
from brown_bridge.features import species_direction
from brown_bridge.reciprocal import (
    DirectionalTimeMLP,
    InequalityAugmentedLagrangian,
    integrate_directional_sde,
    make_reciprocal_batch,
    orthogonal_displacement,
    shared_noise_cycle_loss,
    species_progression_loss,
)


SEED = 107
STEPS = 2_500
BATCH_SIZE = 256
CONSTRAINT_BATCH = 96
CONSTRAINT_EVERY = 8
CONSTRAINT_STEPS = 8
CONSTRAINT_DIRECTIONS = 128
CONSTRAINT_THRESHOLD = 5e-5
CONTENT_THRESHOLD = 4e-5
RECIPROCAL_THRESHOLD = 1.5e-4
AUGMENT_PENALTY = 5_000.0
CONTENT_PENALTY = 50_000.0
RECIPROCAL_PENALTY = 20_000.0
DUAL_WARMUP = 250
LAMBDA_CONTENT = 2.0
LAMBDA_RECIPROCAL = 2.0
SPECIES_WEIGHT = 0.05


def train_augmented(
    split: base.Split,
    plan: Tensor,
    species: Tensor,
    *,
    steps: int,
    threshold: float,
    penalty: float,
    device: torch.device,
    resume: dict[str, object] | None = None,
) -> tuple[DirectionalTimeMLP, DirectionalTimeMLP, list[dict[str, float]], list[dict[str, float]]]:
    torch.manual_seed(SEED)
    velocity = DirectionalTimeMLP(base.FEATURE_DIM, width=256, depth=3, time_frequencies=3).to(device)
    score = DirectionalTimeMLP(base.FEATURE_DIM, width=256, depth=3, time_frequencies=3).to(device)
    optimizer = torch.optim.AdamW(
        [*velocity.parameters(), *score.parameters()], lr=3e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(SEED + 1)
    directions = F.normalize(
        torch.randn(CONSTRAINT_DIRECTIONS, base.FEATURE_DIM, generator=generator), dim=1
    ).to(device)
    constraints = {
        "endpoint": InequalityAugmentedLagrangian(
            threshold=threshold,
            penalty=penalty,
        ),
        "content": InequalityAugmentedLagrangian(
            threshold=CONTENT_THRESHOLD,
            penalty=CONTENT_PENALTY,
        ),
        "reciprocal": InequalityAugmentedLagrangian(
            threshold=RECIPROCAL_THRESHOLD,
            penalty=RECIPROCAL_PENALTY,
        ),
    }
    history: list[dict[str, float]] = [] if resume is None else list(resume["history"])
    dual_history: list[dict[str, float]] = [] if resume is None else list(resume["dual_history"])
    if resume is not None:
        velocity.load_state_dict(resume["velocity"])
        score.load_state_dict(resume["score"])
        last = dual_history[-1]
        for key in constraints:
            constraints[key].multiplier = float(last[f"{key}_multiplier"])
    step_offset = len(history)

    for step in range(1, steps + 1):
        absolute_step = step_offset + step
        dual_active = resume is not None or absolute_step > DUAL_WARMUP
        batch = make_reciprocal_batch(
            plan,
            split.cat_global,
            split.dog_global,
            BATCH_SIZE,
            diffusivity=base.TAU,
            generator=generator,
            device=device,
        )
        predicted_velocity = velocity(batch.state, batch.time, batch.direction)
        predicted_score = score(batch.state, batch.time, batch.direction)
        per_sample = (predicted_velocity - batch.conditional_velocity).square().mean(1)
        flow_loss = (batch.flow_weight * per_sample).sum() / batch.flow_weight.sum()
        gamma = torch.sqrt(
            2.0 * base.TAU * batch.time * (1.0 - batch.time)
        ).clamp_min(1e-6)
        score_loss = F.mse_loss(gamma[:, None] * predicted_score, batch.scaled_score)
        species_loss = species_progression_loss(
            predicted_velocity, batch.direction, species.to(device)
        )
        terminal_value = torch.zeros((), device=device)
        augmented_loss = torch.zeros((), device=device)
        content_loss = torch.zeros((), device=device)
        reciprocal_loss = torch.zeros((), device=device)

        if step % CONSTRAINT_EVERY == 0:
            cat_choice = torch.randint(base.FIT_COUNT, (CONSTRAINT_BATCH,), generator=generator)
            dog_choice = torch.randint(base.FIT_COUNT, (CONSTRAINT_BATCH,), generator=generator)
            cats = split.cat_global[cat_choice].to(device)
            dogs = split.dog_global[dog_choice].to(device)
            forward = integrate_directional_sde(
                velocity,
                score,
                cats,
                direction=1,
                steps=CONSTRAINT_STEPS,
                kappa=0.0,
            )
            backward = integrate_directional_sde(
                velocity,
                score,
                dogs,
                direction=-1,
                steps=CONSTRAINT_STEPS,
                kappa=0.0,
            )
            terminal_value = 0.5 * (
                base.sliced_wasserstein(forward[-1], dogs, directions)
                + base.sliced_wasserstein(backward[-1], cats, directions)
            )
            cat_reciprocal, _, _ = shared_noise_cycle_loss(
                velocity,
                score,
                cats[: base.TERMINAL_BATCH],
                steps=base.TERMINAL_STEPS,
                kappa=base.TRAIN_KAPPA,
                generator=generator,
            )
            dog_reciprocal, _, _ = shared_noise_cycle_loss(
                velocity,
                score,
                dogs[: base.TERMINAL_BATCH],
                steps=base.TERMINAL_STEPS,
                kappa=base.TRAIN_KAPPA,
                generator=generator,
                direction=-1,
            )
            reciprocal_loss = 0.5 * (cat_reciprocal + dog_reciprocal)
            content_loss = 0.5 * (
                orthogonal_displacement(forward[-1], cats, species.to(device))
                + orthogonal_displacement(backward[-1], dogs, species.to(device))
            )
            if dual_active:
                augmented_loss = (
                    constraints["endpoint"].primal_loss(terminal_value)
                    + constraints["content"].primal_loss(content_loss)
                    + constraints["reciprocal"].primal_loss(reciprocal_loss)
                )

        total = (
            flow_loss
            + score_loss
            + SPECIES_WEIGHT * species_loss
            + augmented_loss
            + LAMBDA_CONTENT * CONSTRAINT_EVERY * content_loss
            + LAMBDA_RECIPROCAL * CONSTRAINT_EVERY * reciprocal_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([*velocity.parameters(), *score.parameters()], 1.0)
        optimizer.step()
        scheduler.step()

        if step % CONSTRAINT_EVERY == 0 and dual_active:
            multipliers = {
                "endpoint": constraints["endpoint"].dual_step(terminal_value),
                "content": constraints["content"].dual_step(content_loss),
                "reciprocal": constraints["reciprocal"].dual_step(reciprocal_loss),
            }
            dual_history.append(
                {
                    "step": absolute_step,
                    "endpoint_value": float(terminal_value.detach().cpu()),
                    "content_value": float(content_loss.detach().cpu()),
                    "reciprocal_value": float(reciprocal_loss.detach().cpu()),
                    "endpoint_multiplier": multipliers["endpoint"],
                    "content_multiplier": multipliers["content"],
                    "reciprocal_multiplier": multipliers["reciprocal"],
                }
            )
        record = {
            "total": float(total.detach().cpu()),
            "flow": float(flow_loss.detach().cpu()),
            "score": float(score_loss.detach().cpu()),
            "species": float(species_loss.detach().cpu()),
            "terminal_constraint": float(terminal_value.detach().cpu()),
            "augmented": float(augmented_loss.detach().cpu()),
            "content": float(content_loss.detach().cpu()),
            "reciprocal": float(reciprocal_loss.detach().cpu()),
            "endpoint_multiplier": constraints["endpoint"].multiplier,
            "content_multiplier": constraints["content"].multiplier,
            "reciprocal_multiplier": constraints["reciprocal"].multiplier,
        }
        history.append(record)
        if step == 1 or step % 250 == 0:
            recent = history[-100:]
            mean_flow = sum(item["flow"] for item in recent) / len(recent)
            constrained = dual_history[-1] if dual_history else {
                "endpoint_value": 0.0,
                "endpoint_multiplier": 0.0,
            }
            print(
                f"augmented step={absolute_step:04d} flow={mean_flow:.5f} "
                f"g={constrained['endpoint_value']:.7f} "
                f"dual={constrained['endpoint_multiplier']:.2f}",
                flush=True,
            )
    return velocity.eval(), score.eval(), history, dual_history


def plot_comparison(reference: dict[str, object], augmented: dict[str, object], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    fixed = reference["variants"]["regularized"]["curves"]
    current = augmented["curves"]
    specifications = (
        ("forward_sinkhorn", "Held-out cat→dog alignment", "Sinkhorn divergence"),
        ("forward_dog_probability", "Independent species progression", "P(dog)"),
        ("forward_content_mse", "Content displacement", "orthogonal MSE"),
    )
    for axis, (key, title, ylabel) in zip(axes, specifications):
        axis.plot(fixed["time"], fixed[key], "--", label="fixed λ")
        axis.plot(current["time"], current[key], "-", label="augmented constraint")
        axis.set(title=title, xlabel="time", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.savefig(output / "augmented_diagnostics.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--constraint-threshold", type=float, default=CONSTRAINT_THRESHOLD)
    parser.add_argument("--penalty", type=float, default=AUGMENT_PENALTY)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.CLASSIFIER = args.classifier.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = base.host_device()
    if device.type != "mps":
        raise RuntimeError(f"requested MPS run, but selected device is {device}")
    features = base.extract_features(args.data_root, output, device)
    splits = base.make_splits(features)
    species = species_direction(splits["fit"].cat_global, splits["fit"].dog_global)
    plan, content_cost, coupling = base.content_plan(splits["fit"], species)
    probe = base.fit_probe(splits["validation"])
    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume is not None
        else None
    )
    velocity, score, history, dual_history = train_augmented(
        splits["fit"],
        plan,
        species,
        steps=args.steps,
        threshold=args.constraint_threshold,
        penalty=args.penalty,
        device=device,
        resume=resume,
    )
    metrics, trajectories = base.evaluate_variant(
        velocity,
        score,
        splits["test"],
        species,
        probe,
        device=device,
    )
    reference = json.loads(args.reference_summary.read_text())
    fixed = reference["variants"]["regularized"]
    comparison = {
        "cat_to_dog_sinkhorn_change": metrics["forward"]["sinkhorn_terminal"]
        - fixed["forward"]["sinkhorn_terminal"],
        "dog_to_cat_sinkhorn_change": metrics["backward"]["sinkhorn_terminal"]
        - fixed["backward"]["sinkhorn_terminal"],
        "cat_to_dog_content_ratio": metrics["forward"]["terminal_content_mse"]
        / fixed["forward"]["terminal_content_mse"],
        "shared_noise_cycle_ratio": metrics["shared_noise_cycle_mse_mean"]
        / fixed["shared_noise_cycle_mse_mean"],
    }
    last_dual = dual_history[-1]
    summary: dict[str, object] = {
        "device": str(device),
        "method": "reciprocal SF2M with PHR augmented endpoint inequality",
        "training_steps": len(history),
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "constraints": {
            "metric": "bidirectional 128-projection sliced W2 over 96-sample minibatches",
            "thresholds": {
                "endpoint": args.constraint_threshold,
                "content": CONTENT_THRESHOLD,
                "reciprocal": RECIPROCAL_THRESHOLD,
            },
            "penalties": {
                "endpoint": args.penalty,
                "content": CONTENT_PENALTY,
                "reciprocal": RECIPROCAL_PENALTY,
            },
            "dual_warmup_steps": DUAL_WARMUP,
            "updates": len(dual_history),
            "final_values": {
                key: last_dual[f"{key}_value"]
                for key in ("endpoint", "content", "reciprocal")
            },
            "final_multipliers": {
                key: last_dual[f"{key}_multiplier"]
                for key in ("endpoint", "content", "reciprocal")
            },
            "recent_20_value_means": {
                key: sum(item[f"{key}_value"] for item in dual_history[-20:])
                / min(20, len(dual_history))
                for key in ("endpoint", "content", "reciprocal")
            },
        },
        "regularization": {
            "lambda_content": LAMBDA_CONTENT,
            "lambda_reciprocal": LAMBDA_RECIPROCAL,
        },
        "coupling": coupling,
        "test": metrics,
        "comparison_to_fixed_lambda": comparison,
    }
    torch.save(
        {
            "velocity": velocity.state_dict(),
            "score": score.state_dict(),
            "history": history,
            "dual_history": dual_history,
            "trajectories": trajectories,
            "species_direction": species,
            "fit_plan": plan,
            "fit_content_cost": content_cost,
            "probe": probe.state_dict(),
        },
        output / "augmented_reciprocal_bridge.pt",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_comparison(reference, metrics, output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
