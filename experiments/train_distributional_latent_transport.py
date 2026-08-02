"""Train source-only LightSB and minibatch-OT rectified flow on full AFHQ.

The experiment reuses the strong 128px autoencoder.  It transports an exactly
decodable 8x8x8 low-frequency coordinate and carries the source's complementary
spatial-latent residual through the path.  Neither model receives a dog image at
inference; held-out dogs are used only as a distributional evaluation measure.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.flow_matching import (
    SpatialTimeConditionedField,
    TimeConditionedMLP,
    integrate_flow,
)
from brown_bridge.latent_image_bridge import LatentImageAutoencoder
from brown_bridge.light_sb import (
    CoordinateStandardizer,
    DiagonalLightSB,
    brownian_bridge_path,
    coarse_latent_coordinate,
    expected_minibatch_ot_pairs,
    lift_coarse_coordinate,
)
from brown_bridge.ot import sinkhorn_divergence
from brown_bridge.probes import BinaryLinearProbe
from brown_bridge.stochastic_interpolants import sliced_wasserstein_squared
from render_dino_stargan_bridge import (
    dino_encode,
    make_contact_sheet,
    make_distribution_gif,
    tensor_to_pil,
    unique_minimum_assignment,
)
from render_full_dataset_demo import fit_or_load_afhq_probe, load_models
from train_continuous_vae_renderer import AFHQDomainDataset


@torch.inference_mode()
def encode_coordinates(
    model: LatentImageAutoencoder,
    dataset: AFHQDomainDataset,
    *,
    batch_size: int,
    factor: int,
    device: torch.device,
) -> Tensor:
    batches: list[Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for index, images in enumerate(loader, start=1):
        batches.append(coarse_latent_coordinate(model.encode(images.to(device)), factor=factor).cpu())
        if index % 25 == 0:
            print(f"encoded {min(index * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    return torch.cat(batches)


def coordinate_cache(
    model: LatentImageAutoencoder,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, AFHQDomainDataset]]:
    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=True, augment=False
        )
        for domain in ("cat", "dog")
    }
    names = {domain: [path.name for path in data.paths] for domain, data in datasets.items()}
    if args.coordinate_cache.exists():
        cache = torch.load(args.coordinate_cache, map_location="cpu", weights_only=False)
        if cache.get("names") != names or cache.get("factor") != args.pool_factor:
            raise RuntimeError("latent coordinate cache does not match this filtered dataset")
        return cache["cat"].float(), cache["dog"].float(), datasets
    coordinates = {
        domain: encode_coordinates(
            model,
            dataset,
            batch_size=args.encode_batch_size,
            factor=args.pool_factor,
            device=device,
        )
        for domain, dataset in datasets.items()
    }
    args.coordinate_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**coordinates, "names": names, "factor": args.pool_factor}, args.coordinate_cache)
    return coordinates["cat"], coordinates["dog"], datasets


@torch.inference_mode()
def training_dino_content_features(
    dino: torch.nn.Module,
    datasets: dict[str, AFHQDomainDataset],
    preprocess: BatchPreprocessor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Cache species-orthogonal SSL features aligned with latent-cache rows."""
    names = {
        domain: [path.name for path in dataset.paths]
        for domain, dataset in datasets.items()
    }
    if args.ot_feature_cache.exists():
        saved = torch.load(args.ot_feature_cache, map_location="cpu", weights_only=False)
        if saved.get("names") != names:
            raise RuntimeError("DINO OT-feature cache does not match the latent dataset")
        return saved["cat_content"].float(), saved["dog_content"].float()
    features: dict[str, Tensor] = {}
    for domain, dataset in datasets.items():
        batches: list[Tensor] = []
        loader = DataLoader(
            dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=0
        )
        for index, images in enumerate(loader, start=1):
            global_state, _, _ = dino_encode(dino, images, preprocess, device)
            batches.append(F.normalize(global_state.float(), dim=1))
            if index % 25 == 0:
                print(
                    f"DINO OT features {domain}: "
                    f"{min(index * args.feature_batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
        features[domain] = torch.cat(batches)
    direction = F.normalize(features["dog"].mean(0) - features["cat"].mean(0), dim=0)
    content = {
        domain: F.normalize(
            value - (value @ direction)[:, None] * direction, dim=1
        )
        for domain, value in features.items()
    }
    args.ot_feature_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cat_content": content["cat"],
            "dog_content": content["dog"],
            "species_direction": direction,
            "names": names,
        },
        args.ot_feature_cache,
    )
    return content["cat"], content["dog"]


def make_transport_pairs(
    source: Tensor,
    target: Tensor,
    train_data: dict[str, AFHQDomainDataset],
    dino: torch.nn.Module,
    preprocess: BatchPreprocessor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    source_cost = target_cost = None
    if args.ot_cost_space == "dino":
        source_cost, target_cost = training_dino_content_features(
            dino, train_data, preprocess, args, device
        )
    return expected_minibatch_ot_pairs(
        source,
        target,
        batch_size=args.ot_batch_size,
        batches=args.ot_batches,
        epsilon=args.ot_epsilon,
        sinkhorn_iterations=args.ot_iterations,
        generator=torch.Generator().manual_seed(args.seed + 3),
        source_cost_features=source_cost,
        target_cost_features=target_cost,
        pairing=args.ot_pairing,
    )


def train_lightsb(
    source: Tensor,
    target: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DiagonalLightSB, list[dict[str, float]]]:
    model = DiagonalLightSB(
        source.shape[1],
        components=args.sb_components,
        epsilon=args.sb_epsilon,
        initial_scale=args.sb_initial_scale,
    ).to(device)
    model.initialize_centers(target, generator=torch.Generator().manual_seed(args.seed + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.sb_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sb_steps)
    generator = torch.Generator().manual_seed(args.seed + 2)
    history: list[dict[str, float]] = []
    for step in range(1, args.sb_steps + 1):
        x = source[torch.randint(len(source), (args.batch_size,), generator=generator)].to(device)
        y = target[torch.randint(len(target), (args.batch_size,), generator=generator)].to(device)
        base = model.objective(x, y)
        scale_regularizer = (
            model.log_scale - math.log(args.sb_initial_scale)
        ).square().mean()
        average_weight = model.conditional_logits(x).softmax(1).mean(0)
        balance_regularizer = (
            average_weight * (average_weight.clamp_min(1e-8).log() + math.log(args.sb_components))
        ).sum()
        objective_per_dimension = base / source.shape[1]
        loss = (
            objective_per_dimension
            + args.lambda_scale * scale_regularizer
            + args.lambda_balance * balance_regularizer
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.sb_steps:
            record = {
                "step": float(step),
                "objective": float(base.detach()),
                "objective_per_dimension": float(objective_per_dimension.detach()),
                "scale_regularizer": float(scale_regularizer.detach()),
                "balance_regularizer": float(balance_regularizer.detach()),
                "scale_min": float(model.scale.detach().min()),
                "scale_max": float(model.scale.detach().max()),
            }
            history.append(record)
            print("LightSB " + json.dumps(record), flush=True)
    return model.eval(), history


def train_rectified_flow(
    paired_source: Tensor,
    paired_target: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    model = make_flow_model(paired_source.shape[1], args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.flow_learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.flow_steps)
    generator = torch.Generator().manual_seed(args.seed + 4)
    history: list[dict[str, float]] = []
    for step in range(1, args.flow_steps + 1):
        index = torch.randint(len(paired_source), (args.batch_size,), generator=generator)
        left, right = paired_source[index].to(device), paired_target[index].to(device)
        time = torch.rand(args.batch_size, generator=generator).to(device)
        state = (1.0 - time[:, None]) * left + time[:, None] * right
        target_velocity = right - left
        prediction = model(state, time)
        loss = F.mse_loss(prediction, target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.flow_steps:
            record = {"step": float(step), "flow_matching_mse": float(loss.detach())}
            history.append(record)
            print("rectified flow " + json.dumps(record), flush=True)
    return model.eval(), history


def make_flow_model(feature_dim: int, args: argparse.Namespace) -> torch.nn.Module:
    if args.flow_architecture == "spatial":
        expected = args.latent_channels * args.latent_spatial_size**2
        if feature_dim != expected:
            raise ValueError(
                f"spatial flow expects {expected} features, received {feature_dim}"
            )
        return SpatialTimeConditionedField(
            args.latent_channels,
            args.latent_spatial_size,
            base_channels=args.spatial_flow_width,
            time_dim=args.spatial_time_dim,
        )
    return TimeConditionedMLP(
        feature_dim, width=args.flow_width, depth=args.flow_depth
    )


def _differentiable_flow_endpoint(
    model: torch.nn.Module,
    initial: Tensor,
    *,
    steps: int,
    reverse: bool = False,
) -> Tensor:
    """Full-interval Heun rollout used only by terminal-law fine-tuning."""
    state = initial
    start = 1.0 if reverse else 0.0
    step_size = (-1.0 if reverse else 1.0) / steps
    for index in range(steps):
        physical_time = start + index * step_size
        time = torch.full(
            (len(state),), physical_time, device=state.device, dtype=state.dtype
        ).clamp(0.005, 0.995)
        velocity = model(state, time)
        proposal = state + step_size * velocity
        next_time = torch.full(
            (len(state),), physical_time + step_size,
            device=state.device, dtype=state.dtype,
        ).clamp(0.005, 0.995)
        next_velocity = model(proposal, next_time)
        state = state + 0.5 * step_size * (velocity + next_velocity)
    return state


def _transport_directions(
    source: Tensor,
    target: Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Mix global domain and sample-difference projections for sliced OT."""
    if count < 2:
        raise ValueError("endpoint projections must be at least two")
    mean_delta = target.mean(0) - source.mean(0)
    pooled_variance = 0.5 * (
        source.var(0, unbiased=False) + target.var(0, unbiased=False)
    )
    fisher_delta = mean_delta / pooled_variance.clamp_min(0.05)
    source_index = torch.randint(len(source), (count - 2,), generator=generator)
    target_index = torch.randint(len(target), (count - 2,), generator=generator)
    paired_delta = target[target_index] - source[source_index]
    directions = torch.cat([mean_delta[None], fisher_delta[None], paired_delta], dim=0)
    return F.normalize(directions, dim=1)


def _semantic_features_from_coordinate(
    coordinate: Tensor,
    autoencoder: LatentImageAutoencoder,
    standardizer: CoordinateStandardizer,
    dino: torch.nn.Module,
    preprocess: BatchPreprocessor,
) -> Tensor:
    """Decode complete latents and expose differentiable normalized DINO features."""
    raw = standardizer.inverse(coordinate).reshape(
        len(coordinate), autoencoder.latent_channels, 32, 32
    )
    images = autoencoder.decode(raw)
    normalized = preprocess(images, training=False)
    features, _ = dino.encode(normalized)
    return F.normalize(features.float(), dim=1)


def finetune_endpoint_law(
    model: torch.nn.Module,
    source: Tensor,
    target: Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    autoencoder: LatentImageAutoencoder,
    standardizer: CoordinateStandardizer,
    dino: torch.nn.Module,
    preprocess: BatchPreprocessor,
    species_direction: Tensor,
) -> tuple[torch.nn.Module, list[dict[str, float | str]]]:
    """Coerce both terminal laws without retraining the image decoder.

    The frozen pre-coercion field acts as a local trust region.  The new term is
    bidirectional sliced W2 in decoder-visible DINO space, evaluated after a
    short differentiable probability-flow rollout.
    """
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    model = model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.endpoint_learning_rate, weight_decay=1e-5
    )
    generator = torch.Generator().manual_seed(args.seed + 20)
    if args.endpoint_space == "dino":
        direction_sample_count = min(128, len(source), len(target))
        semantic_source_batches: list[Tensor] = []
        semantic_target_batches: list[Tensor] = []
        with torch.no_grad():
            for chunk in source[:direction_sample_count].split(args.endpoint_batch_size):
                semantic_source_batches.append(
                    _semantic_features_from_coordinate(
                        chunk.to(device), autoencoder, standardizer, dino, preprocess
                    ).cpu()
                )
            for chunk in target[:direction_sample_count].split(args.endpoint_batch_size):
                semantic_target_batches.append(
                    _semantic_features_from_coordinate(
                        chunk.to(device), autoencoder, standardizer, dino, preprocess
                    ).cpu()
                )
        semantic_source = torch.cat(semantic_source_batches)
        semantic_target = torch.cat(semantic_target_batches)
        directions = _transport_directions(
            semantic_source,
            semantic_target,
            args.endpoint_projections,
            generator=generator,
        )
        directions[0] = F.normalize(species_direction.float(), dim=0)
        directions = directions.to(device)
    else:
        directions = _transport_directions(
            source,
            target,
            args.endpoint_projections,
            generator=generator,
        ).to(device)
    history: list[dict[str, float]] = []

    evaluation_count = min(128, len(source), len(target))
    evaluation_source = source[:evaluation_count].to(device)
    evaluation_target = target[:evaluation_count].to(device)
    with torch.no_grad():
        baseline_forward = _differentiable_flow_endpoint(
            teacher, evaluation_source,
            steps=max(args.endpoint_rollout_steps, 16),
        )
        if args.endpoint_space == "dino":
            baseline_coordinate = _semantic_features_from_coordinate(
                baseline_forward, autoencoder, standardizer, dino, preprocess
            )
            evaluation_target_coordinate = _semantic_features_from_coordinate(
                evaluation_target, autoencoder, standardizer, dino, preprocess
            )
        else:
            baseline_coordinate = baseline_forward
            evaluation_target_coordinate = evaluation_target
        baseline_w2 = sliced_wasserstein_squared(
            baseline_coordinate, evaluation_target_coordinate, directions=directions
        )

    for step in range(1, args.endpoint_coercion_steps + 1):
        source_index = torch.randint(
            len(source), (args.endpoint_batch_size,), generator=generator
        )
        target_index = torch.randint(
            len(target), (args.endpoint_batch_size,), generator=generator
        )
        left = source[source_index].to(device)
        right = target[target_index].to(device)
        forward = _differentiable_flow_endpoint(
            model, left, steps=args.endpoint_rollout_steps
        )
        backward = _differentiable_flow_endpoint(
            model, right, steps=args.endpoint_rollout_steps, reverse=True
        )
        if args.endpoint_space == "dino":
            forward_features = _semantic_features_from_coordinate(
                forward, autoencoder, standardizer, dino, preprocess
            )
            backward_features = _semantic_features_from_coordinate(
                backward, autoencoder, standardizer, dino, preprocess
            )
            with torch.no_grad():
                left_features = _semantic_features_from_coordinate(
                    left, autoencoder, standardizer, dino, preprocess
                )
                right_features = _semantic_features_from_coordinate(
                    right, autoencoder, standardizer, dino, preprocess
                )
            endpoint_loss = sliced_wasserstein_squared(
                forward_features, right_features, directions=directions
            ) + sliced_wasserstein_squared(
                backward_features, left_features, directions=directions
            )
            domain_direction = F.normalize(species_direction.to(device), dim=0)
            species_loss = (
                sliced_wasserstein_squared(
                    forward_features, right_features, directions=domain_direction[None]
                )
                + sliced_wasserstein_squared(
                    backward_features, left_features, directions=domain_direction[None]
                )
            )
            forward_content = F.normalize(
                forward_features
                - (forward_features @ domain_direction)[:, None] * domain_direction,
                dim=1,
            )
            backward_content = F.normalize(
                backward_features
                - (backward_features @ domain_direction)[:, None] * domain_direction,
                dim=1,
            )
            left_content = F.normalize(
                left_features - (left_features @ domain_direction)[:, None] * domain_direction,
                dim=1,
            )
            right_content = F.normalize(
                right_features - (right_features @ domain_direction)[:, None] * domain_direction,
                dim=1,
            )
            content_loss = 0.5 * (
                1.0 - (forward_content * left_content).sum(1).mean()
                + 1.0 - (backward_content * right_content).sum(1).mean()
            )
        else:
            endpoint_loss = sliced_wasserstein_squared(
                forward, right, directions=directions
            ) + sliced_wasserstein_squared(
                backward, left, directions=directions
            )
            content_loss = endpoint_loss.new_zeros(())
            species_loss = endpoint_loss.new_zeros(())

        time = torch.rand(args.endpoint_batch_size, generator=generator).to(device)
        state = (1.0 - time[:, None]) * left + time[:, None] * right
        with torch.no_grad():
            teacher_velocity = teacher(state, time)
        trust_region = F.mse_loss(model(state, time), teacher_velocity)
        progress = step / args.endpoint_coercion_steps
        endpoint_weight = args.lambda_endpoint * 0.5 * (
            1.0 - math.cos(math.pi * progress)
        )
        loss = (
            endpoint_weight * endpoint_loss
            + args.lambda_species * species_loss
            + args.lambda_teacher * trust_region
            + args.lambda_content * content_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.endpoint_log_every == 0 or step == args.endpoint_coercion_steps:
            record = {
                "step": float(step),
                "endpoint_sliced_w2": float(endpoint_loss.detach()),
                "teacher_velocity_mse": float(trust_region.detach()),
                "content_loss": float(content_loss.detach()),
                "species_sliced_w2": float(species_loss.detach()),
                "endpoint_weight": float(endpoint_weight),
            }
            history.append(record)
            print("endpoint coercion " + json.dumps(record), flush=True)

    model.eval()
    with torch.no_grad():
        coerced_forward = _differentiable_flow_endpoint(
            model, evaluation_source,
            steps=max(args.endpoint_rollout_steps, 16),
        )
        if args.endpoint_space == "dino":
            coerced_coordinate = _semantic_features_from_coordinate(
                coerced_forward, autoencoder, standardizer, dino, preprocess
            )
        else:
            coerced_coordinate = coerced_forward
        coerced_w2 = sliced_wasserstein_squared(
            coerced_coordinate, evaluation_target_coordinate, directions=directions
        )
    history.append(
        {
            "step": float(args.endpoint_coercion_steps),
            "heldout_baseline_sliced_w2": float(baseline_w2),
            "heldout_coerced_sliced_w2": float(coerced_w2),
            "endpoint_space": args.endpoint_space,
        }
    )
    print("endpoint coercion evaluation " + json.dumps(history[-1]), flush=True)
    return model, history


@torch.inference_mode()
def decode_coordinate_path(
    model: LatentImageAutoencoder,
    source_latent: Tensor,
    standardized_path: Tensor,
    standardizer: CoordinateStandardizer,
    *,
    factor: int,
    decode_batch_size: int,
    residual_weight: Tensor | None = None,
) -> Tensor:
    time_count, batch, dimension = standardized_path.shape
    coordinate = standardizer.inverse(standardized_path).reshape(time_count * batch, dimension)
    repeated_source = source_latent[None].expand(time_count, -1, -1, -1, -1).reshape(
        time_count * batch, *source_latent.shape[1:]
    )
    if residual_weight is None:
        residual_weight = torch.ones(time_count)
    if residual_weight.shape != (time_count,):
        raise ValueError("residual_weight must have one value per path time")
    weight = residual_weight[:, None].expand(time_count, batch).reshape(
        time_count * batch, 1, 1, 1
    )
    latent = lift_coarse_coordinate(
        repeated_source, coordinate, factor=factor, residual_weight=weight
    )
    decoded = [model.decode(chunk) for chunk in latent.split(decode_batch_size)]
    return torch.cat(decoded).reshape(time_count, batch, 3, 128, 128).transpose(0, 1).cpu()


def path_metrics(
    decoded: Tensor,
    dino: torch.nn.Module,
    probe: BinaryLinearProbe,
    preprocess: object,
    device: torch.device,
    *,
    direction: int = 1,
) -> tuple[dict[str, float | list[list[float]]], Tensor]:
    global_batches: list[Tensor] = []
    for images in decoded.flatten(0, 1).split(64):
        global_state, _, _ = dino_encode(dino, images, preprocess, device)
        global_batches.append(global_state)
    global_state = torch.cat(global_batches)
    features = global_state.reshape(decoded.shape[0], decoded.shape[1], -1)
    probability = probe.probabilities(features.flatten(0, 1)).reshape(decoded.shape[:2])
    delta = probability[:, 1:] - probability[:, :-1]
    oriented_delta = direction * delta
    speed = torch.linalg.vector_norm(features[:, 1:] - features[:, :-1], dim=2)
    total_variation = delta.abs().sum(1)
    net = (probability[:, -1] - probability[:, 0]).abs()
    return {
        "source_dog_probability_mean": float(probability[:, 0].mean()),
        "endpoint_dog_probability_mean": float(probability[:, -1].mean()),
        "semantic_monotone_step_fraction": float((oriented_delta >= -0.01).float().mean()),
        "semantic_monotone_path_fraction": float((oriented_delta >= -0.01).all(1).float().mean()),
        "semantic_total_variation_excess_mean": float((total_variation - net).mean()),
        "mean_dino_feature_step": float(speed.mean()),
        "coefficient_of_variation_dino_speed": float(
            speed.std(unbiased=False) / speed.mean().clamp_min(1e-8)
        ),
        "mean_adjacent_frame_l1": float((decoded[:, 1:] - decoded[:, :-1]).abs().mean()),
        "path_dog_probability": probability.tolist(),
    }, features


def endpoint_distribution_metrics(
    source_features: Tensor,
    generated_features: Tensor,
    target_features: Tensor,
    species_direction: Tensor,
    *,
    target_name: str,
) -> dict[str, float]:
    divergence, _ = sinkhorn_divergence(
        generated_features, target_features, epsilon=0.05, cost="cosine", max_iterations=300
    )
    direction = F.normalize(species_direction.float(), dim=0)
    source_content = F.normalize(source_features - (source_features @ direction)[:, None] * direction, dim=1)
    generated_content = F.normalize(
        generated_features - (generated_features @ direction)[:, None] * direction, dim=1
    )
    generated_distance = torch.pdist(F.normalize(generated_features, dim=1)).mean()
    target_distance = torch.pdist(F.normalize(target_features, dim=1)).mean()
    return {
        f"dino_sinkhorn_divergence_to_{target_name}": float(divergence),
        "source_content_cosine_mean": float((source_content * generated_content).sum(1).mean()),
        f"endpoint_diversity_ratio_to_{target_name}": float(
            generated_distance / target_distance.clamp_min(1e-8)
        ),
    }


@torch.inference_mode()
def integrate_reverse_flow(
    model: torch.nn.Module,
    initial: Tensor,
    *,
    macro_steps: int,
    microsteps_per_macro: int,
) -> Tensor:
    """Integrate the same probability-flow law from physical time one to zero."""
    total_steps = macro_steps * microsteps_per_macro
    step_size = -1.0 / total_steps
    state = initial
    path = [state]
    for index in range(total_steps):
        time_value = 1.0 + index * step_size
        time = torch.full((len(state),), time_value, device=state.device).clamp(0.005, 0.995)
        velocity = model(state, time)
        proposal = state + step_size * velocity
        next_time = torch.full(
            (len(state),), time_value + step_size, device=state.device
        ).clamp(0.005, 0.995)
        state = state + 0.5 * step_size * (velocity + model(proposal, next_time))
        path.append(state)
    return torch.stack(path)


def make_domain_contact_sheet(
    sources: list[Image.Image],
    paths: list[list[Image.Image]],
    references: list[Image.Image],
    output: Path,
    *,
    source_name: str,
    target_name: str,
    clock_label: str,
    reference_label: str | None = None,
) -> None:
    """Render a domain-neutral trajectory sheet (the legacy helper says cat→dog)."""
    tile, header = 256, 42
    labels = [f"{source_name}: t=0"]
    labels.extend(f"t={index / max(len(paths[0]) - 1, 1):.2f}" for index in range(len(paths[0])))
    labels.append(reference_label or f"held-out {target_name} ref")
    columns = len(labels)
    canvas = Image.new("RGB", (columns * tile, header + len(paths) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * tile + 8, 7), label, fill="black")
    draw.text((8, header - 15), clock_label, fill=(70, 70, 70))
    for row, (source, frames, reference) in enumerate(zip(sources, paths, references)):
        y = header + row * tile
        canvas.paste(source, (0, y))
        for column, frame in enumerate(frames, start=1):
            canvas.paste(frame, (column * tile, y))
        canvas.paste(reference, ((columns - 1) * tile, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def save_diagnostic_plot(probabilities: dict[str, Tensor], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for name, values in probabilities.items():
        mean = values.mean(0)
        time = torch.linspace(0, 1, len(mean))
        axis.plot(time, mean, label=name)
        axis.fill_between(time, values.quantile(0.1, dim=0), values.quantile(0.9, dim=0), alpha=0.15)
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.set(xlabel="time", ylabel="AFHQ DINO probe P(dog)", ylim=(-0.02, 1.02))
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def pil_paths(decoded: Tensor) -> list[list[Image.Image]]:
    return [
        [
            tensor_to_pil(frame.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
            for frame in path
        ]
        for path in decoded
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this full image experiment requires Apple MPS")
    device = torch.device("mps")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    autoencoder, dino, prior_summary = load_models(args, device)
    preprocess = BatchPreprocessor(64)
    probe, species_direction, probe_metrics = fit_or_load_afhq_probe(dino, args, device)
    raw_cat, raw_dog, train_data = coordinate_cache(autoencoder, args, device)
    training_checkpoint = (
        args.transport_checkpoint
        if args.transport_checkpoint is not None
        else args.output_dir / "trained_transport_checkpoint.pt"
    )
    if args.evaluate_only:
        saved = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
        standardizer = CoordinateStandardizer.from_state_dict(saved["standardizer"])
        lightsb = DiagonalLightSB(
            raw_cat.shape[1], components=args.sb_components, epsilon=args.sb_epsilon,
            initial_scale=args.sb_initial_scale,
        )
        lightsb.load_state_dict(saved["lightsb"])
        lightsb = lightsb.eval().to(device)
        sb_history = saved["sb_history"]
        if args.flow_architecture == "spatial":
            spatial_checkpoint = args.output_dir / "trained_spatial_flow.pt"
            flow = make_flow_model(raw_cat.shape[1], args)
            if spatial_checkpoint.exists() and not args.force_flow_retrain:
                spatial_saved = torch.load(
                    spatial_checkpoint, map_location="cpu", weights_only=False
                )
                flow.load_state_dict(spatial_saved["flow"])
                flow_history = spatial_saved["flow_history"]
                flow = flow.eval().to(device)
                print(f"loaded spatial flow from {spatial_checkpoint}", flush=True)
            else:
                cat = standardizer.transform(raw_cat)
                dog = standardizer.transform(raw_dog)
                paired_cat, paired_dog = make_transport_pairs(
                    cat, dog, train_data, dino, preprocess, args, device
                )
                flow, flow_history = train_rectified_flow(
                    paired_cat, paired_dog, args, device
                )
                torch.save(
                    {
                        "flow": flow.state_dict(),
                        "flow_history": flow_history,
                        "architecture": "spatial",
                    },
                    spatial_checkpoint,
                )
                print(f"saved spatial flow to {spatial_checkpoint}", flush=True)
        else:
            flow = make_flow_model(raw_cat.shape[1], args)
            flow.load_state_dict(saved["flow"])
            flow = flow.eval().to(device)
            flow_history = saved["flow_history"]
        print(f"loaded trained transports from {training_checkpoint}", flush=True)
    else:
        standardizer = CoordinateStandardizer.fit(torch.cat([raw_cat, raw_dog]))
        cat = standardizer.transform(raw_cat)
        dog = standardizer.transform(raw_dog)
        lightsb, sb_history = train_lightsb(cat, dog, args, device)
        paired_cat, paired_dog = make_transport_pairs(
            cat, dog, train_data, dino, preprocess, args, device
        )
        flow, flow_history = train_rectified_flow(paired_cat, paired_dog, args, device)
        torch.save(
            {
                "lightsb": lightsb.state_dict(),
                "flow": flow.state_dict(),
                "standardizer": standardizer.state_dict(),
                "sb_history": sb_history,
                "flow_history": flow_history,
            },
            training_checkpoint,
        )
        print(f"saved trained transports to {training_checkpoint}", flush=True)

    endpoint_history: list[dict[str, float | str]] = []
    if args.endpoint_coercion_steps:
        standardized_cat = standardizer.transform(raw_cat)
        standardized_dog = standardizer.transform(raw_dog)
        flow, endpoint_history = finetune_endpoint_law(
            flow,
            standardized_cat,
            standardized_dog,
            args,
            device,
            autoencoder=autoencoder,
            standardizer=standardizer,
            dino=dino,
            preprocess=preprocess,
            species_direction=species_direction,
        )
        torch.save(
            {
                "flow": flow.state_dict(),
                "standardizer": standardizer.state_dict(),
                "history": endpoint_history,
            },
            args.output_dir / "endpoint_coerced_flow.pt",
        )

    test = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    generator = torch.Generator().manual_seed(args.seed + 5)
    eval_cat_index = torch.randperm(len(test["cat"]), generator=generator)[: args.eval_examples]
    eval_dog_index = torch.randperm(len(test["dog"]), generator=generator)[: args.eval_examples]
    eval_cats = torch.stack([test["cat"][int(index)] for index in eval_cat_index])
    eval_dogs = torch.stack([test["dog"][int(index)] for index in eval_dog_index])
    source_latent = torch.cat(
        [autoencoder.encode(images.to(device)) for images in eval_cats.split(args.decode_batch_size)]
    )
    target_latent = torch.cat(
        [autoencoder.encode(images.to(device)) for images in eval_dogs.split(args.decode_batch_size)]
    )
    source_coordinate = standardizer.transform(
        coarse_latent_coordinate(source_latent, factor=args.pool_factor)
    )
    target_coordinate = standardizer.transform(
        coarse_latent_coordinate(target_latent, factor=args.pool_factor)
    )
    with torch.no_grad():
        sb_mean_endpoint = lightsb.conditional_mean(source_coordinate)
        sb_sample_endpoint = lightsb.sample_endpoint(
            source_coordinate, generator=torch.Generator().manual_seed(args.seed + 6)
        )
        display_time = torch.linspace(0, 1, args.display_steps)
        sb_mean_path = (
            (1 - display_time[:, None, None].to(device)) * source_coordinate[None, : args.examples]
            + display_time[:, None, None].to(device) * sb_mean_endpoint[None, : args.examples]
        )
        sb_sample_path = brownian_bridge_path(
            source_coordinate[: args.examples],
            sb_sample_endpoint[: args.examples],
            display_time,
            epsilon=args.sb_epsilon,
            generator=torch.Generator().manual_seed(args.seed + 7),
        )
        _, flow_dense = integrate_flow(
            flow,
            source_coordinate,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.flow_microsteps,
            method="heun",
            geometry="ambient",
        )
        flow_endpoint = flow_dense[-1]
        flow_path = flow_dense[:: args.flow_microsteps, : args.examples]
        reverse_flow_dense = integrate_reverse_flow(
            flow,
            target_coordinate,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.flow_microsteps,
        )
        reverse_flow_endpoint = reverse_flow_dense[-1]
        reverse_flow_path = reverse_flow_dense[:: args.flow_microsteps, : args.examples]
        residual_weight = args.residual_end + (1.0 - args.residual_end) * (
            1.0 - display_time
        ).pow(args.residual_power)
        endpoint_residual_weight = torch.tensor([args.residual_end])

    decoded = {
        "LightSB mean": decode_coordinate_path(
            autoencoder, source_latent[: args.examples], sb_mean_path, standardizer,
            factor=args.pool_factor, decode_batch_size=args.decode_batch_size,
            residual_weight=residual_weight,
        ),
        "LightSB SDE": decode_coordinate_path(
            autoencoder, source_latent[: args.examples], sb_sample_path, standardizer,
            factor=args.pool_factor, decode_batch_size=args.decode_batch_size,
            residual_weight=residual_weight,
        ),
        "OT rectified flow": decode_coordinate_path(
            autoencoder, source_latent[: args.examples], flow_path, standardizer,
            factor=args.pool_factor, decode_batch_size=args.decode_batch_size,
            residual_weight=residual_weight,
        ),
    }
    reverse_decoded = decode_coordinate_path(
        autoencoder,
        target_latent[: args.examples],
        reverse_flow_path,
        standardizer,
        factor=args.pool_factor,
        decode_batch_size=args.decode_batch_size,
        residual_weight=residual_weight,
    )
    print("decoded dense display paths", flush=True)
    endpoint_decoded = {
        "LightSB mean": decode_coordinate_path(
            autoencoder, source_latent, sb_mean_endpoint[None], standardizer,
            factor=args.pool_factor, decode_batch_size=args.decode_batch_size,
            residual_weight=endpoint_residual_weight,
        )[:, 0],
        "LightSB SDE": decode_coordinate_path(
            autoencoder, source_latent, sb_sample_endpoint[None], standardizer,
            factor=args.pool_factor, decode_batch_size=args.decode_batch_size,
            residual_weight=endpoint_residual_weight,
        )[:, 0],
        "OT rectified flow": decode_coordinate_path(
            autoencoder, source_latent, flow_endpoint[None], standardizer,
            factor=args.pool_factor, decode_batch_size=args.decode_batch_size,
            residual_weight=endpoint_residual_weight,
        )[:, 0],
    }
    reverse_endpoint_decoded = decode_coordinate_path(
        autoencoder,
        target_latent,
        reverse_flow_endpoint[None],
        standardizer,
        factor=args.pool_factor,
        decode_batch_size=args.decode_batch_size,
        residual_weight=endpoint_residual_weight,
    )[:, 0]
    print("decoded held-out endpoint populations", flush=True)
    metrics: dict[str, dict[str, object]] = {}
    probability_tensors: dict[str, Tensor] = {}
    for name, path in decoded.items():
        path_result, path_features = path_metrics(path, dino, probe, preprocess, device)
        metrics[name] = path_result
        probability_tensors[name] = torch.tensor(path_result["path_dog_probability"])
        print(f"evaluated dense path: {name}", flush=True)
    reverse_path_result, _ = path_metrics(
        reverse_decoded, dino, probe, preprocess, device, direction=-1
    )
    probability_tensors["OT rectified flow: dog to cat"] = torch.tensor(
        reverse_path_result["path_dog_probability"]
    )

    source_features, _, _ = dino_encode(dino, eval_cats.cpu(), preprocess, device)
    dog_features, _, _ = dino_encode(dino, eval_dogs, preprocess, device)
    for name in decoded:
        endpoint_features, _, _ = dino_encode(
            dino, endpoint_decoded[name], preprocess, device
        )
        metrics[name].update(
            endpoint_distribution_metrics(
                source_features,
                endpoint_features,
                dog_features,
                species_direction,
                target_name="dogs",
            )
        )
        print(f"evaluated endpoint law: {name}", flush=True)
    reverse_endpoint_features, _, _ = dino_encode(
        dino, reverse_endpoint_decoded, preprocess, device
    )
    reverse_path_result.update(
        endpoint_distribution_metrics(
            dog_features,
            reverse_endpoint_features,
            source_features,
            species_direction,
            target_name="cats",
        )
    )
    metrics["OT rectified flow: dog to cat"] = reverse_path_result
    print("evaluated reverse endpoint law: OT rectified flow", flush=True)
    save_diagnostic_plot(probability_tensors, args.output_dir / "semantic_clock_comparison.png")

    # A dog is displayed only as a nearest content reference; it is never an inference input.
    direction = F.normalize(species_direction.float(), dim=0)
    cat_content = F.normalize(source_features - (source_features @ direction)[:, None] * direction, dim=1)
    dog_content = F.normalize(dog_features - (dog_features @ direction)[:, None] * direction, dim=1)
    reference_pool = min(len(dog_content), max(args.examples, 10))
    assignment = unique_minimum_assignment(
        1.0 - cat_content[: args.examples] @ dog_content[:reference_pool].T
    )
    reverse_reference_pool = min(len(cat_content), max(args.examples, 10))
    reverse_assignment = unique_minimum_assignment(
        1.0 - dog_content[: args.examples] @ cat_content[:reverse_reference_pool].T
    )
    references = [
        tensor_to_pil(eval_dogs[index].mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
        for index in assignment[: args.examples]
    ]
    sources = [
        tensor_to_pil(image.cpu().mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
        for image in eval_cats[: args.examples]
    ]
    reverse_references = [
        tensor_to_pil(eval_cats[index].mul(2).sub(1)).resize(
            (256, 256), Image.Resampling.LANCZOS
        )
        for index in reverse_assignment[: args.examples]
    ]
    reverse_sources = [
        tensor_to_pil(image.cpu().mul(2).sub(1)).resize(
            (256, 256), Image.Resampling.LANCZOS
        )
        for image in eval_dogs[: args.examples]
    ]
    sheet_index = torch.linspace(0, args.display_steps - 1, args.sheet_steps).round().long().tolist()
    for name, path in decoded.items():
        slug = name.lower().replace(" ", "_")
        paths = pil_paths(path[: args.examples])
        make_contact_sheet(
            sources,
            [[frames[index] for index in sheet_index] for frames in paths],
            references,
            args.output_dir / f"{slug}_contact_sheet.png",
            clock_label=name,
        )
        make_distribution_gif(
            paths,
            args.output_dir / f"{slug}.gif",
            clock_label=name,
            duration_ms=args.frame_duration_ms,
        )
    reverse_paths = pil_paths(reverse_decoded)
    make_domain_contact_sheet(
        reverse_sources,
        [[frames[index] for index in sheet_index] for frames in reverse_paths],
        reverse_references,
        args.output_dir / "ot_rectified_flow_dog_to_cat_contact_sheet.png",
        source_name="dog",
        target_name="cat",
        clock_label="same OT probability flow, integrated t=1 to t=0",
    )

    residual_description = (
        "none: the transport coordinate is the complete spatial latent"
        if args.pool_factor == 1
        else (
            f"source residual z-U(Pz) follows weight "
            f"{args.residual_end}+(1-{args.residual_end})(1-t)^{args.residual_power}"
        )
    )
    summary: dict[str, object] = {
        "claim": "one bidirectional distributional cat↔dog probability flow in a decodable image latent",
        "endpoint_conditioning": (
            "No paired endpoint is supplied. Displayed target-domain images are held-out, "
            "evaluation-only content references."
        ),
        "coordinate": {
            "raw_latent": list(source_latent.shape[1:]),
            "pool_factor": args.pool_factor,
            "dimension": raw_cat.shape[1],
            "fine_residual": residual_description,
        },
        "train_counts": {domain: len(data) for domain, data in train_data.items()},
        "test_counts": {domain: len(data) for domain, data in test.items()},
        "cat_test_indices": eval_cat_index.tolist(),
        "dog_test_indices": eval_dog_index.tolist(),
        "reference_assignment": assignment,
        "reverse_reference_assignment": reverse_assignment,
        "afhq_probe_test_metrics": probe_metrics,
        "metrics": metrics,
        "endpoint_coercion_history": endpoint_history,
        "training_quarantine_verified": prior_summary["quarantine_verified"],
        "config": vars(args) | {"afhq_root": str(args.afhq_root), "output_dir": str(args.output_dir),
                                 "coordinate_cache": str(args.coordinate_cache),
                                 "ot_feature_cache": str(args.ot_feature_cache),
                                 "probe_cache": str(args.probe_cache),
                                 "model_checkpoint": str(args.model_checkpoint), "dino_checkpoint": str(args.dino_checkpoint),
                                 "transport_checkpoint": (
                                     str(args.transport_checkpoint)
                                     if args.transport_checkpoint is not None else None
                                 )},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {
            "lightsb": lightsb.state_dict(),
            "flow": flow.state_dict(),
            "standardizer": standardizer.state_dict(),
            "sb_history": sb_history,
            "flow_history": flow_history,
            "endpoint_coercion_history": endpoint_history,
            "summary": summary,
        },
        args.output_dir / "distributional_transport.pt",
    )
    torch.save(
        {**decoded, "OT rectified flow: dog to cat": reverse_decoded},
        args.output_dir / "trajectories.pt",
    )
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument("--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt")
    parser.add_argument("--model-checkpoint", type=Path, default=project / "outputs/full_dataset_latent_bridge/full_dataset_latent_bridge.pt")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/distributional_latent_transport_full")
    parser.add_argument("--transport-checkpoint", type=Path)
    parser.add_argument("--coordinate-cache", type=Path)
    parser.add_argument("--ot-feature-cache", type=Path)
    parser.add_argument("--probe-cache", type=Path, default=project / "outputs/full_dataset_latent_bridge_demo/afhq_dino_linear_probe.pt")
    parser.add_argument("--pool-factor", type=int, default=1)
    parser.add_argument("--residual-end", type=float, default=1.0)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sb-components", type=int, default=10)
    parser.add_argument("--sb-epsilon", type=float, default=0.1)
    parser.add_argument("--sb-initial-scale", type=float, default=0.1)
    parser.add_argument("--sb-learning-rate", type=float, default=1e-3)
    parser.add_argument("--sb-steps", type=int, default=10000)
    parser.add_argument("--lambda-scale", type=float, default=1e-2)
    parser.add_argument("--lambda-balance", type=float, default=1.0)
    parser.add_argument("--ot-batch-size", type=int, default=128)
    parser.add_argument("--ot-batches", type=int, default=256)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iterations", type=int, default=40)
    parser.add_argument("--ot-cost-space", choices=("latent", "dino"), default="latent")
    parser.add_argument("--ot-pairing", choices=("sample", "rounded"), default="sample")
    parser.add_argument("--flow-width", type=int, default=256)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument(
        "--flow-architecture", choices=("mlp", "spatial"), default="mlp"
    )
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--latent-spatial-size", type=int, default=32)
    parser.add_argument("--spatial-flow-width", type=int, default=48)
    parser.add_argument("--spatial-time-dim", type=int, default=128)
    parser.add_argument("--force-flow-retrain", action="store_true")
    parser.add_argument("--flow-learning-rate", type=float, default=3e-4)
    parser.add_argument("--flow-steps", type=int, default=6000)
    parser.add_argument("--flow-microsteps", type=int, default=2)
    parser.add_argument("--endpoint-coercion-steps", type=int, default=0)
    parser.add_argument("--endpoint-space", choices=("latent", "dino"), default="dino")
    parser.add_argument("--endpoint-rollout-steps", type=int, default=8)
    parser.add_argument("--endpoint-batch-size", type=int, default=32)
    parser.add_argument("--endpoint-projections", type=int, default=48)
    parser.add_argument("--endpoint-learning-rate", type=float, default=2e-5)
    parser.add_argument("--lambda-endpoint", type=float, default=1.0)
    parser.add_argument("--lambda-species", type=float, default=4.0)
    parser.add_argument("--lambda-teacher", type=float, default=0.05)
    parser.add_argument("--lambda-content", type=float, default=0.1)
    parser.add_argument("--endpoint-log-every", type=int, default=25)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--frame-duration-ms", type=int, default=160)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.coordinate_cache is None:
        args.coordinate_cache = args.output_dir / "train_coarse_latents.pt"
    if args.ot_feature_cache is None:
        args.ot_feature_cache = args.output_dir / "train_dino_content_features.pt"
    if args.pool_factor < 1 or 32 % args.pool_factor:
        parser.error("--pool-factor must divide the 32x32 image latent")
    if args.flow_architecture == "spatial" and args.pool_factor != 1:
        parser.error("spatial flow currently requires --pool-factor 1")
    if min(
        args.latent_channels,
        args.latent_spatial_size,
        args.spatial_flow_width,
        args.spatial_time_dim,
    ) < 1:
        parser.error("spatial flow dimensions must be positive")
    if not 0 <= args.residual_end <= 1 or args.residual_power <= 0:
        parser.error("residual-end must be in [0,1] and residual-power positive")
    if not 2 <= args.sheet_steps <= args.display_steps:
        parser.error("invalid sheet/display step counts")
    if args.eval_examples < 2 or not 1 <= args.examples <= args.eval_examples:
        parser.error("examples must be positive and no larger than eval-examples")
    if args.endpoint_coercion_steps < 0 or args.endpoint_rollout_steps < 2:
        parser.error("invalid endpoint coercion or rollout step count")
    if args.endpoint_batch_size < 2 or args.endpoint_projections < 2:
        parser.error("endpoint batch size and projections must be at least two")
    if args.endpoint_space == "dino" and args.pool_factor != 1:
        parser.error("DINO endpoint coercion currently requires --pool-factor 1")
    if (
        args.endpoint_learning_rate <= 0
        or min(
            args.lambda_endpoint,
            args.lambda_species,
            args.lambda_teacher,
            args.lambda_content,
        )
        < 0
    ):
        parser.error("endpoint learning rate and regularization weights must be nonnegative")
    return args


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(run(configuration), indent=2))
