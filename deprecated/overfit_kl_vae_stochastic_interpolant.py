"""Deliberately memorize one AFHQ OT pair to validate the reciprocal SDE."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.spatial_stochastic import (
    integrate_probability_flow,
    integrate_reciprocal_sde,
)
from train_bidirectional_kl_vae_diffusion import (
    decode_batches,
    decode_path,
    latent_cache,
    load_vae,
    pair_latents,
    pil_images,
)
from train_bidirectional_kl_vae_rectified_flow import make_path_gif
from train_bidirectional_kl_vae_stochastic_interpolant import train_field
from train_distributional_latent_transport import (
    make_domain_contact_sheet,
    pil_paths,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    device = torch.device("mps")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_vae(args, device)
    latents, content, _ = latent_cache(vae, args, device)
    paired_cat, paired_dog = pair_latents(latents, content, args)
    if not 0 <= args.pair_index < len(paired_cat):
        raise ValueError("pair-index is out of range")
    cat = paired_cat[args.pair_index : args.pair_index + 1]
    dog = paired_dog[args.pair_index : args.pair_index + 1]
    field, history = train_field(cat, dog, args, device)
    torch.save(
        {
            "model": field.state_dict(),
            "history": history,
            "overfit_pair_index": args.pair_index,
        },
        args.checkpoint,
    )

    samples = args.samples
    cat_flat = cat.flatten(1).to(device).repeat(samples, 1)
    dog_flat = dog.flatten(1).to(device).repeat(samples, 1)
    paths_flat = {
        "sde_cat_to_dog": integrate_reciprocal_sde(
            field,
            cat_flat,
            interpolant_epsilon=args.interpolant_epsilon,
            peak_diffusion_rate=args.peak_diffusion_rate,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
        ),
        "sde_dog_to_cat": integrate_reciprocal_sde(
            field,
            dog_flat,
            interpolant_epsilon=args.interpolant_epsilon,
            peak_diffusion_rate=args.peak_diffusion_rate,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
            reverse=True,
        ),
        "ode_cat_to_dog": integrate_probability_flow(
            field,
            cat_flat,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
        ),
        "ode_dog_to_cat": integrate_probability_flow(
            field,
            dog_flat,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
            reverse=True,
        ),
    }
    latent_shape = cat.shape[1:]
    paths = {
        name: value.reshape(*value.shape[:2], *latent_shape)
        for name, value in paths_flat.items()
    }
    decoded = {
        name: decode_path(
            vae, value, batch_size=args.vae_batch_size, device=device
        )
        for name, value in paths.items()
    }
    endpoint_images = decode_batches(
        vae,
        torch.cat([cat, dog]).to(device),
        batch_size=args.vae_batch_size,
        device=device,
    )
    cat_image, dog_image = endpoint_images[:1], endpoint_images[1:]
    sheet_indices = torch.linspace(
        0, args.display_steps - 1, args.sheet_steps
    ).round().long().tolist()
    for method in ("sde", "ode"):
        for direction_name, source, target, source_name, target_name in (
            ("cat_to_dog", cat_image, dog_image, "memorized cat", "memorized dog"),
            ("dog_to_cat", dog_image, cat_image, "memorized dog", "memorized cat"),
        ):
            name = f"{method}_{direction_name}"
            path_images = pil_paths(decoded[name])
            make_domain_contact_sheet(
                pil_images(source.repeat(samples, 1, 1, 1)),
                [[row[index] for index in sheet_indices] for row in path_images],
                pil_images(target.repeat(samples, 1, 1, 1)),
                args.output_dir / f"overfit_{name}.png",
                source_name=source_name,
                target_name=target_name,
                clock_label=(
                    "overfit reciprocal Nelson SDE"
                    if method == "sde"
                    else "overfit probability-flow ODE diagnostic"
                ),
            )
            if method == "sde":
                make_path_gif(
                    path_images,
                    args.output_dir / f"overfit_{name}.gif",
                    source_name=source_name,
                    target_name=target_name,
                )

    metrics: dict[str, dict[str, float]] = {}
    for name, value in paths.items():
        target = dog.to(device) if name.endswith("cat_to_dog") else cat.to(device)
        source = cat.to(device) if name.endswith("cat_to_dog") else dog.to(device)
        endpoint_error = (value[-1] - target).square().flatten(1).mean(1)
        metrics[name] = {
            "source_latent_mse": float(
                (value[0] - source).square().flatten(1).mean(1).mean()
            ),
            "endpoint_latent_mse_mean": float(endpoint_error.mean()),
            "endpoint_latent_mse_best": float(endpoint_error.min()),
            "endpoint_image_l1_mean": float(
                F.l1_loss(
                    decoded[name][:, -1],
                    (dog_image if name.endswith("cat_to_dog") else cat_image).repeat(
                        samples, 1, 1, 1
                    ),
                )
            ),
            "mean_adjacent_frame_l1": float(
                (decoded[name][:, 1:] - decoded[name][:, :-1]).abs().mean()
            ),
        }
    summary = {
        "purpose": "single-pair overfit hypothesis test, not generalization evidence",
        "overfit_pair_index": args.pair_index,
        "exclude_this_pair_from_full_training": True,
        "same_current_and_score_both_directions": True,
        "metrics": metrics,
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {name: value.cpu() for name, value in paths.items()},
        args.output_dir / "overfit_trajectories.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument("--ot-feature-cache", type=Path, default=project / "outputs/distributional_latent_transport_full/train_dino_content_features.pt")
    parser.add_argument("--latent-cache", type=Path, default=project / "outputs/bidirectional_kl_vae_diffusion/train_kl_vae_latents.pt")
    parser.add_argument("--teacher-checkpoint", type=Path, default=project / "outputs/bidirectional_kl_vae_rectified_flow/rectified_flow.pt")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/overfit_kl_vae_stochastic_interpolant")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--interpolant-epsilon", type=float, default=0.08)
    parser.add_argument("--minimum-time", type=float, default=0.01)
    parser.add_argument("--velocity-boundary-floor", type=float, default=0.05)
    parser.add_argument("--peak-diffusion-rate", type=float, default=0.04)
    parser.add_argument("--lambda-score", type=float, default=0.5)
    parser.add_argument("--lambda-teacher", type=float, default=0.01)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.1)
    parser.add_argument("--lambda-spatial-gradient", type=float, default=0.05)
    parser.add_argument("--score-warmup-steps", type=int, default=500)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--ot-batch-size", type=int, default=64)
    parser.add_argument("--ot-epsilon", type=float, default=0.03)
    parser.add_argument("--ot-iterations", type=int, default=75)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "overfit_stochastic_interpolant.pt"
    return args


if __name__ == "__main__":
    run(parse_args())
