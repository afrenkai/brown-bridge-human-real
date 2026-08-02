"""Reciprocal cat/dog SDE on DINO-coupled frozen KL-VAE latents.

The model learns the current velocity and the scaled marginal score of a
Brownian stochastic interpolant.  One physical field defines the forward and
backward Nelson drifts; no target image is supplied at held-out inference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.flow_matching import SpatialTimeConditionedField
from brown_bridge.spatial_stochastic import (
    SpatialVelocityScoreField,
    bridge_score_scale,
    integrate_probability_flow,
    integrate_reciprocal_sde,
)
from render_dino_stargan_bridge import dino_encode
from render_full_dataset_demo import fit_or_load_afhq_probe
from train_bidirectional_kl_vae_diffusion import (
    decode_batches,
    decode_path,
    encode_batches,
    latent_cache,
    load_dino,
    load_vae,
    pair_latents,
    pil_images,
    reconstruction_sheet,
)
from train_bidirectional_kl_vae_rectified_flow import make_path_gif, sample_bridge_time
from train_continuous_vae_renderer import AFHQDomainDataset
from train_distributional_latent_transport import (
    endpoint_distribution_metrics,
    make_domain_contact_sheet,
    path_metrics,
    pil_paths,
)


def make_teacher(
    args: argparse.Namespace,
    latent_shape: torch.Size,
    device: torch.device,
) -> SpatialTimeConditionedField:
    saved = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = SpatialTimeConditionedField(
        latent_shape[0],
        latent_shape[-1],
        base_channels=args.model_width,
        time_dim=args.time_dim,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
    )
    teacher.load_state_dict(saved["model"])
    return teacher.eval().requires_grad_(False).to(device)


def warm_start_velocity(
    field: SpatialVelocityScoreField,
    teacher: SpatialTimeConditionedField,
) -> None:
    """Copy the rectified-flow trunk and velocity head; zero the score head."""
    student = field.backbone.state_dict()
    source = teacher.state_dict()
    for name, value in source.items():
        if name not in student:
            continue
        if student[name].shape == value.shape:
            student[name].copy_(value)
        elif name in {"output.2.weight", "output.2.bias"}:
            student[name].zero_()
            student[name][: len(value)].copy_(value)
        else:
            raise RuntimeError(f"cannot warm-start parameter {name}")
    field.backbone.load_state_dict(student)


def train_field(
    cat: Tensor,
    dog: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SpatialVelocityScoreField, list[dict[str, float]]]:
    latent_shape = cat.shape[1:]
    field = SpatialVelocityScoreField(
        latent_shape[0],
        latent_shape[-1],
        base_channels=args.model_width,
        time_dim=args.time_dim,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
    ).to(device)
    teacher = make_teacher(args, latent_shape, device)
    warm_start_velocity(field, teacher)
    optimizer = torch.optim.AdamW(
        field.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator().manual_seed(args.seed + 30)
    cat_flat, dog_flat = cat.flatten(1), dog.flatten(1)
    channels, spatial_size = latent_shape[0], latent_shape[-1]
    history: list[dict[str, float]] = []

    for step in range(1, args.steps + 1):
        index = torch.randint(len(cat_flat), (args.batch_size,), generator=generator)
        source = cat_flat[index].to(device)
        target = dog_flat[index].to(device)
        time = sample_bridge_time(args.batch_size, generator).clamp(
            args.minimum_time, 1.0 - args.minimum_time
        ).to(device)
        noise = torch.randn(source.shape, generator=generator).to(device)
        gamma = bridge_score_scale(
            time,
            interpolant_epsilon=args.interpolant_epsilon,
            minimum_time=args.minimum_time,
        )
        weight = time[:, None]
        mean = (1.0 - weight) * source + weight * target
        state = mean + gamma[:, None] * noise
        target_velocity = (
            target
            - source
            + args.interpolant_epsilon
            * (1.0 - 2.0 * time[:, None])
            * noise
            / (2.0 * gamma[:, None])
        )
        target_scaled_score = -noise
        velocity, scaled_score = field(state, time)
        flow_weight = gamma.square() / (
            gamma.square() + args.velocity_boundary_floor**2
        )
        flow_mse = (
            flow_weight
            * (velocity - target_velocity).square().mean(1)
        ).sum() / flow_weight.sum().clamp_min(1e-8)
        score_mse = F.mse_loss(scaled_score, target_scaled_score)

        velocity_spatial = velocity.reshape(-1, channels, spatial_size, spatial_size)
        target_spatial = target_velocity.reshape_as(velocity_spatial)
        low_frequency = F.mse_loss(
            F.avg_pool2d(velocity_spatial, 4), F.avg_pool2d(target_spatial, 4)
        )
        spatial_gradient = 0.5 * (
            F.smooth_l1_loss(
                velocity_spatial[..., 1:] - velocity_spatial[..., :-1],
                target_spatial[..., 1:] - target_spatial[..., :-1],
            )
            + F.smooth_l1_loss(
                velocity_spatial[..., 1:, :] - velocity_spatial[..., :-1, :],
                target_spatial[..., 1:, :] - target_spatial[..., :-1, :],
            )
        )
        with torch.no_grad():
            teacher_velocity = teacher(state, time)
        teacher_mse = F.mse_loss(velocity, teacher_velocity)
        score_weight = args.lambda_score * min(1.0, step / args.score_warmup_steps)
        loss = (
            flow_mse
            + score_weight * score_mse
            + args.lambda_low_frequency * low_frequency
            + args.lambda_spatial_gradient * spatial_gradient
            + args.lambda_teacher * teacher_mse
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": float(step),
                "flow_mse": float(flow_mse.detach()),
                "scaled_score_mse": float(score_mse.detach()),
                "teacher_velocity_mse": float(teacher_mse.detach()),
                "low_frequency_mse": float(low_frequency.detach()),
                "spatial_gradient_l1": float(spatial_gradient.detach()),
                "score_weight": float(score_weight),
            }
            history.append(record)
            print("KL stochastic interpolant " + json.dumps(record), flush=True)
    return field.eval(), history


def load_field(
    args: argparse.Namespace, latent_shape: torch.Size, device: torch.device
) -> tuple[SpatialVelocityScoreField, list[dict[str, float]]]:
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    field = SpatialVelocityScoreField(
        latent_shape[0],
        latent_shape[-1],
        base_channels=args.model_width,
        time_dim=args.time_dim,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
    )
    field.load_state_dict(saved["model"])
    return field.eval().to(device), saved["history"]


def compact_path_metrics(
    decoded: Tensor,
    dino: torch.nn.Module,
    probe: torch.nn.Module,
    preprocess: object,
    device: torch.device,
    *,
    direction: int,
) -> tuple[dict[str, object], Tensor]:
    metrics, _ = path_metrics(
        decoded, dino, probe, preprocess, device, direction=direction
    )
    probability = torch.tensor(metrics.pop("path_dog_probability"))
    endpoint = probability[:, -1]
    metrics["endpoint_classified_as_target_fraction"] = float(
        ((endpoint >= 0.5) if direction > 0 else (endpoint < 0.5)).float().mean()
    )
    metrics["endpoint_dog_probability_quantiles"] = torch.quantile(
        endpoint, torch.tensor([0.1, 0.5, 0.9])
    ).tolist()
    return metrics, probability


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    device = torch.device("mps")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_vae(args, device)
    dino = load_dino(args, device)
    preprocess = BatchPreprocessor(64)
    probe, species_direction, probe_test = fit_or_load_afhq_probe(dino, args, device)
    latents, content, _ = latent_cache(vae, args, device)
    paired_cat, paired_dog = pair_latents(latents, content, args)
    excluded_pair_indices = sorted(set(args.exclude_pair_indices))
    if any(index < 0 or index >= len(paired_cat) for index in excluded_pair_indices):
        raise ValueError("exclude-pair-indices contains an invalid pair index")
    if excluded_pair_indices:
        keep = torch.ones(len(paired_cat), dtype=torch.bool)
        keep[excluded_pair_indices] = False
        paired_cat, paired_dog = paired_cat[keep], paired_dog[keep]
    if args.evaluate_only:
        field, history = load_field(args, paired_cat.shape[1:], device)
    else:
        field, history = train_field(paired_cat, paired_dog, args, device)
        torch.save({"model": field.state_dict(), "history": history}, args.checkpoint)

    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    generator = torch.Generator().manual_seed(args.seed + 40)
    indices = {
        domain: torch.randperm(len(dataset), generator=generator)[: args.eval_examples]
        for domain, dataset in datasets.items()
    }
    images = {
        domain: torch.stack([dataset[int(index)] for index in indices[domain]])
        for domain, dataset in datasets.items()
    }
    encoded = {
        domain: encode_batches(
            vae, image, batch_size=args.vae_batch_size, device=device
        ).to(device)
        for domain, image in images.items()
    }
    reconstruction_latent = torch.cat([encoded["cat"][:3], encoded["dog"][:3]])
    reconstruction_original = torch.cat([images["cat"][:3], images["dog"][:3]])
    reconstructed = decode_batches(
        vae, reconstruction_latent, batch_size=args.vae_batch_size, device=device
    )
    reconstruction_sheet(
        reconstruction_original,
        reconstructed,
        args.output_dir / "kl_vae_reconstructions.png",
    )

    flat = {domain: value.flatten(1) for domain, value in encoded.items()}
    paths_flat = {
        "sde_cat_to_dog": integrate_reciprocal_sde(
            field,
            flat["cat"],
            interpolant_epsilon=args.interpolant_epsilon,
            peak_diffusion_rate=args.peak_diffusion_rate,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
        ),
        "sde_dog_to_cat": integrate_reciprocal_sde(
            field,
            flat["dog"],
            interpolant_epsilon=args.interpolant_epsilon,
            peak_diffusion_rate=args.peak_diffusion_rate,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
            reverse=True,
        ),
        "ode_cat_to_dog": integrate_probability_flow(
            field,
            flat["cat"],
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
        ),
        "ode_dog_to_cat": integrate_probability_flow(
            field,
            flat["dog"],
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
            reverse=True,
        ),
    }
    latent_shape = encoded["cat"].shape[1:]
    paths = {
        name: value.reshape(*value.shape[:2], *latent_shape)
        for name, value in paths_flat.items()
    }
    decoded_display = {
        name: decode_path(
            vae,
            value[:, : args.examples],
            batch_size=args.vae_batch_size,
            device=device,
        )
        for name, value in paths.items()
    }
    decoded_endpoints = {
        name: decode_batches(
            vae, value[-1], batch_size=args.vae_batch_size, device=device
        )
        for name, value in paths.items()
    }
    real_features = {
        domain: dino_encode(dino, value, preprocess, device)[0]
        for domain, value in images.items()
    }
    metrics: dict[str, dict[str, object]] = {}
    probabilities: dict[str, Tensor] = {}
    for method in ("sde", "ode"):
        for direction_name, source_domain, target_domain, direction in (
            ("cat_to_dog", "cat", "dog", 1),
            ("dog_to_cat", "dog", "cat", -1),
        ):
            name = f"{method}_{direction_name}"
            path_result, probability = compact_path_metrics(
                decoded_display[name],
                dino,
                probe,
                preprocess,
                device,
                direction=direction,
            )
            probabilities[name] = probability
            endpoint_features = dino_encode(
                dino, decoded_endpoints[name], preprocess, device
            )[0]
            endpoint_probability = probe.probabilities(endpoint_features)
            path_result.update(
                {
                    "all_heldout_source_dog_probability_mean": float(
                        probe.probabilities(real_features[source_domain]).mean()
                    ),
                    "all_heldout_endpoint_dog_probability_mean": float(
                        endpoint_probability.mean()
                    ),
                    "all_heldout_endpoint_classified_as_target_fraction": float(
                        (
                            (endpoint_probability >= 0.5)
                            if direction > 0
                            else (endpoint_probability < 0.5)
                        )
                        .float()
                        .mean()
                    ),
                }
            )
            path_result.update(
                endpoint_distribution_metrics(
                    real_features[source_domain],
                    endpoint_features,
                    real_features[target_domain],
                    species_direction,
                    target_name=f"{target_domain}s",
                )
            )
            metrics[name] = path_result

    sheet_indices = torch.linspace(
        0, args.display_steps - 1, args.sheet_steps
    ).round().long().tolist()
    for method in ("sde", "ode"):
        for direction_name, source_domain, target_domain in (
            ("cat_to_dog", "cat", "dog"),
            ("dog_to_cat", "dog", "cat"),
        ):
            name = f"{method}_{direction_name}"
            path_images = pil_paths(decoded_display[name])
            make_domain_contact_sheet(
                pil_images(images[source_domain][: args.examples]),
                [[row[index] for index in sheet_indices] for row in path_images],
                pil_images(images[target_domain][: args.examples]),
                args.output_dir / f"kl_interpolant_{name}.png",
                source_name=source_domain,
                target_name=target_domain,
                clock_label=(
                    "reciprocal endpoint-vanishing Nelson SDE"
                    if method == "sde"
                    else "zero-noise probability-flow ODE diagnostic"
                ),
            )
            if method == "sde":
                make_path_gif(
                    path_images,
                    args.output_dir / f"kl_interpolant_{name}.gif",
                    source_name=source_domain,
                    target_name=target_domain,
                )

    summary = {
        "method": "DINO-OT KL-VAE stochastic interpolant with U-ViT current and score",
        "primary_sampler": "reciprocal Nelson SDE with endpoint-vanishing diffusion rate",
        "same_current_and_score_both_directions": True,
        "inference_uses_target": False,
        "training_counts": {
            "cats": len(latents["cat"]),
            "dogs": len(latents["dog"]),
            "ot_pairs": len(paired_cat),
            "excluded_overfit_pair_indices": excluded_pair_indices,
        },
        "evaluation": {
            "heldout_paths_per_direction": args.eval_examples,
            "displayed_full_paths_per_direction": args.examples,
            "heldout_selection": "random without replacement; seed recorded in config",
        },
        "heldout_vae_reconstruction_l1": float(
            F.l1_loss(reconstructed, reconstruction_original)
        ),
        "metrics": metrics,
        "probe_test": probe_test,
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(probabilities, args.output_dir / "display_path_dog_probabilities.pt")
    torch.save(
        {name: value[:, : args.examples].cpu() for name, value in paths.items()},
        args.output_dir / "display_trajectories.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument("--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt")
    parser.add_argument("--ot-feature-cache", type=Path, default=project / "outputs/distributional_latent_transport_full/train_dino_content_features.pt")
    parser.add_argument("--probe-cache", type=Path, default=project / "outputs/full_dataset_latent_bridge_demo/afhq_dino_linear_probe.pt")
    parser.add_argument("--latent-cache", type=Path, default=project / "outputs/bidirectional_kl_vae_diffusion/train_kl_vae_latents.pt")
    parser.add_argument("--teacher-checkpoint", type=Path, default=project / "outputs/bidirectional_kl_vae_rectified_flow/rectified_flow.pt")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/bidirectional_kl_vae_stochastic_interpolant")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--interpolant-epsilon", type=float, default=0.08)
    parser.add_argument("--minimum-time", type=float, default=0.01)
    parser.add_argument("--velocity-boundary-floor", type=float, default=0.05)
    parser.add_argument("--peak-diffusion-rate", type=float, default=0.04)
    parser.add_argument("--lambda-score", type=float, default=0.15)
    parser.add_argument("--lambda-teacher", type=float, default=0.10)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.10)
    parser.add_argument("--lambda-spatial-gradient", type=float, default=0.05)
    parser.add_argument("--score-warmup-steps", type=int, default=1000)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--ot-batch-size", type=int, default=64)
    parser.add_argument("--ot-epsilon", type=float, default=0.03)
    parser.add_argument("--ot-iterations", type=int, default=75)
    parser.add_argument("--exclude-pair-indices", type=int, nargs="*", default=[0])
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "stochastic_interpolant.pt"
    if not 2 <= args.sheet_steps <= args.display_steps:
        parser.error("invalid display step counts")
    return args


if __name__ == "__main__":
    run(parse_args())
