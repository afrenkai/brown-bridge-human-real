"""Bidirectional rectified flow on frozen KL-VAE AFHQ coordinates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.flow_matching import SpatialTimeConditionedField, integrate_flow
from render_dino_stargan_bridge import dino_encode
from render_full_dataset_demo import fit_or_load_afhq_probe
from train_bidirectional_kl_vae_diffusion import (
    decode_path,
    encode_batches,
    latent_cache,
    load_dino,
    load_vae,
    pair_latents,
    pil_images,
    reconstruction_sheet,
)
from train_continuous_vae_renderer import AFHQDomainDataset
from train_distributional_latent_transport import (
    endpoint_distribution_metrics,
    make_domain_contact_sheet,
    path_metrics,
    pil_paths,
)


def sample_bridge_time(batch_size: int, generator: torch.Generator) -> Tensor:
    """Uniform interior time plus endpoint-biased samples for terminal fidelity."""
    time = torch.rand(batch_size, generator=generator)
    selector = torch.rand(batch_size, generator=generator)
    left = torch.rand(batch_size, generator=generator).square().mul(0.25)
    right = 1.0 - torch.rand(batch_size, generator=generator).square().mul(0.25)
    return torch.where(selector < 0.2, left, torch.where(selector > 0.8, right, time))


def make_path_gif(
    paths: list[list[Image.Image]],
    output: Path,
    *,
    source_name: str,
    target_name: str,
    duration_ms: int = 180,
) -> None:
    """Animate independent held-out trajectories without showing target pairs."""
    if not paths or any(len(path) != len(paths[0]) for path in paths):
        raise ValueError("paths must be nonempty and have a common time grid")
    tile, header = 192, 36
    frames: list[Image.Image] = []
    for step in range(len(paths[0])):
        time = step / max(len(paths[0]) - 1, 1)
        canvas = Image.new("RGB", (len(paths) * tile, header + tile), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 11),
            f"Held-out {source_name} → {target_name}; source-only ODE time t={time:.2f}",
            fill="black",
        )
        for column, path in enumerate(paths):
            canvas.paste(
                path[step].resize((tile, tile), Image.Resampling.LANCZOS),
                (column * tile, header),
            )
        frames.append(canvas)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def train_flow(
    cat: Tensor,
    dog: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SpatialTimeConditionedField, list[dict[str, float]]]:
    channels, spatial_size = cat.shape[1], cat.shape[-1]
    model = SpatialTimeConditionedField(
        channels,
        spatial_size,
        base_channels=args.model_width,
        time_dim=args.time_dim,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    generator = torch.Generator().manual_seed(args.seed + 10)
    history: list[dict[str, float]] = []
    cat_flat, dog_flat = cat.flatten(1), dog.flatten(1)
    for step in range(1, args.steps + 1):
        index = torch.randint(len(cat_flat), (args.batch_size,), generator=generator)
        source = cat_flat[index].to(device)
        target = dog_flat[index].to(device)
        time = sample_bridge_time(args.batch_size, generator).to(device)
        weight = time[:, None]
        state = (1.0 - weight) * source + weight * target
        target_velocity = target - source
        prediction = model(state, time)
        pointwise = F.mse_loss(prediction, target_velocity)
        predicted_spatial = prediction.reshape(
            -1, channels, spatial_size, spatial_size
        )
        target_spatial = target_velocity.reshape_as(predicted_spatial)
        low_frequency = F.mse_loss(
            F.avg_pool2d(predicted_spatial, 4),
            F.avg_pool2d(target_spatial, 4),
        )
        pred_dx = predicted_spatial[..., 1:] - predicted_spatial[..., :-1]
        target_dx = target_spatial[..., 1:] - target_spatial[..., :-1]
        pred_dy = predicted_spatial[..., 1:, :] - predicted_spatial[..., :-1, :]
        target_dy = target_spatial[..., 1:, :] - target_spatial[..., :-1, :]
        spatial_gradient = 0.5 * (
            F.smooth_l1_loss(pred_dx, target_dx)
            + F.smooth_l1_loss(pred_dy, target_dy)
        )
        loss = (
            pointwise
            + args.lambda_low_frequency * low_frequency
            + args.lambda_spatial_gradient * spatial_gradient
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": float(step),
                "flow_mse": float(pointwise.detach()),
                "low_frequency_mse": float(low_frequency.detach()),
                "spatial_gradient_l1": float(spatial_gradient.detach()),
            }
            history.append(record)
            print("KL rectified flow " + json.dumps(record), flush=True)
    return model.eval(), history


def load_flow(args: argparse.Namespace, latent_shape: torch.Size, device: torch.device):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SpatialTimeConditionedField(
        latent_shape[0],
        latent_shape[-1],
        base_channels=args.model_width,
        time_dim=args.time_dim,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
    )
    model.load_state_dict(saved["model"])
    return model.eval().to(device), saved["history"]


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
    if args.evaluate_only:
        model, history = load_flow(args, paired_cat.shape[1:], device)
    else:
        model, history = train_flow(paired_cat, paired_dog, args, device)
        torch.save({"model": model.state_dict(), "history": history}, args.checkpoint)

    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    generator = torch.Generator().manual_seed(args.seed + 20)
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
    reconstruction = torch.cat([encoded["cat"][:3], encoded["dog"][:3]])
    reconstruction_images = torch.cat([images["cat"][:3], images["dog"][:3]])
    reconstructed = torch.cat(
        [vae.decode(chunk) for chunk in reconstruction.split(args.vae_batch_size)]
    ).cpu()
    reconstruction_sheet(
        reconstruction_images,
        reconstructed,
        args.output_dir / "kl_vae_reconstructions.png",
    )

    _, forward_flat = integrate_flow(
        model,
        encoded["cat"].flatten(1),
        macro_steps=args.display_steps - 1,
        microsteps_per_macro=args.microsteps,
        method=args.integrator,
        geometry="ambient",
    )
    _, backward_flat = integrate_flow(
        model,
        encoded["dog"].flatten(1),
        macro_steps=args.display_steps - 1,
        microsteps_per_macro=args.microsteps,
        method=args.integrator,
        geometry="ambient",
        reverse=True,
    )
    latent_shape = encoded["cat"].shape[1:]
    forward_path = forward_flat.reshape(*forward_flat.shape[:2], *latent_shape)
    backward_path = backward_flat.reshape(*backward_flat.shape[:2], *latent_shape)
    decoded_paths = {
        "cat_to_dog": decode_path(
            vae, forward_path, batch_size=args.vae_batch_size, device=device
        ),
        "dog_to_cat": decode_path(
            vae, backward_path, batch_size=args.vae_batch_size, device=device
        ),
    }
    forward_metrics, _ = path_metrics(
        decoded_paths["cat_to_dog"],
        dino,
        probe,
        preprocess,
        device,
        direction=1,
    )
    backward_metrics, _ = path_metrics(
        decoded_paths["dog_to_cat"],
        dino,
        probe,
        preprocess,
        device,
        direction=-1,
    )
    path_probabilities = {
        "cat_to_dog": torch.tensor(forward_metrics.pop("path_dog_probability")),
        "dog_to_cat": torch.tensor(backward_metrics.pop("path_dog_probability")),
    }
    forward_endpoint = path_probabilities["cat_to_dog"][:, -1]
    backward_endpoint = path_probabilities["dog_to_cat"][:, -1]
    forward_metrics.update(
        {
            "endpoint_classified_as_target_fraction": float(
                (forward_endpoint >= 0.5).float().mean()
            ),
            "endpoint_dog_probability_quantiles": torch.quantile(
                forward_endpoint, torch.tensor([0.1, 0.5, 0.9])
            ).tolist(),
        }
    )
    backward_metrics.update(
        {
            "endpoint_classified_as_target_fraction": float(
                (backward_endpoint < 0.5).float().mean()
            ),
            "endpoint_dog_probability_quantiles": torch.quantile(
                backward_endpoint, torch.tensor([0.1, 0.5, 0.9])
            ).tolist(),
        }
    )
    real_features = {
        domain: dino_encode(dino, value, preprocess, device)[0]
        for domain, value in images.items()
    }
    endpoint_images = {
        "dog": decoded_paths["cat_to_dog"][:, -1],
        "cat": decoded_paths["dog_to_cat"][:, -1],
    }
    endpoint_features = {
        domain: dino_encode(dino, value, preprocess, device)[0]
        for domain, value in endpoint_images.items()
    }
    forward_metrics.update(
        endpoint_distribution_metrics(
            real_features["cat"], endpoint_features["dog"], real_features["dog"],
            species_direction, target_name="dogs"
        )
    )
    backward_metrics.update(
        endpoint_distribution_metrics(
            real_features["dog"], endpoint_features["cat"], real_features["cat"],
            species_direction, target_name="cats"
        )
    )

    dense_steps = decoded_paths["cat_to_dog"].shape[1]
    sheet_indices = torch.linspace(0, dense_steps - 1, args.sheet_steps).round().long().tolist()
    for name, source_domain, target_domain in (
        ("cat_to_dog", "cat", "dog"),
        ("dog_to_cat", "dog", "cat"),
    ):
        make_domain_contact_sheet(
            pil_images(images[source_domain][: args.examples]),
            [[row[index] for index in sheet_indices] for row in pil_paths(decoded_paths[name][: args.examples])],
            pil_images(images[target_domain][: args.examples]),
            args.output_dir / f"kl_rectified_flow_{name}.png",
            source_name=source_domain,
            target_name=target_domain,
            clock_label="same invertible KL-latent U-ViT vector field",
        )
        make_path_gif(
            pil_paths(decoded_paths[name][: args.examples]),
            args.output_dir / f"kl_rectified_flow_{name}.gif",
            source_name=source_domain,
            target_name=target_domain,
        )

    summary = {
        "method": "bidirectional OT rectified flow on frozen KL-VAE latents",
        "same_field_both_directions": True,
        "inference_uses_target": False,
        "training_counts": {
            "cats": len(latents["cat"]), "dogs": len(latents["dog"]), "ot_pairs": len(paired_cat)
        },
        "evaluation": {
            "heldout_paths_per_direction": args.eval_examples,
            "displayed_paths_per_direction": args.examples,
            "heldout_selection": "random without replacement; seed recorded in config",
            "target_images_used_only_for_distributional_evaluation": True,
        },
        "heldout_vae_reconstruction_l1": float(F.l1_loss(reconstructed, reconstruction_images)),
        "metrics": {"cat_to_dog": forward_metrics, "dog_to_cat": backward_metrics},
        "probe_test": probe_test,
        "history": history,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(path_probabilities, args.output_dir / "path_dog_probabilities.pt")
    torch.save(decoded_paths, args.output_dir / "trajectories.pt")
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
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/bidirectional_kl_vae_rectified_flow")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.1)
    parser.add_argument("--lambda-spatial-gradient", type=float, default=0.05)
    parser.add_argument("--ot-batch-size", type=int, default=64)
    parser.add_argument("--ot-epsilon", type=float, default=0.03)
    parser.add_argument("--ot-iterations", type=int, default=75)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--microsteps", type=int, default=2)
    parser.add_argument("--integrator", choices=("euler", "heun", "rk4"), default="heun")
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "rectified_flow.pt"
    return args


if __name__ == "__main__":
    run(parse_args())
