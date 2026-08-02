"""Render exact KL-VAE chords induced by alternative AFHQ OT couplings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_bidirectional_kl_vae_diffusion import (
    decode_batches,
    latent_cache,
    load_vae,
    pair_latents,
    pil_images,
)
from train_distributional_latent_transport import make_domain_contact_sheet, pil_paths


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("mps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_vae(args, device)
    latents, global_content, _ = latent_cache(vae, args, device)
    if args.cost == "spatial":
        saved = torch.load(
            args.spatial_ot_feature_cache, map_location="cpu", weights_only=False
        )
        content = {domain: saved[domain].float() for domain in ("cat", "dog")}
    else:
        content = global_content
    paired_cat, paired_dog = pair_latents(latents, content, args)
    generator = torch.Generator().manual_seed(args.seed + 70)
    indices = torch.randperm(len(paired_cat), generator=generator)[: args.examples]
    cat = paired_cat[indices].to(device)
    dog = paired_dog[indices].to(device)
    time = torch.linspace(0.0, 1.0, args.display_steps, device=device)
    path = (
        (1.0 - time[:, None, None, None, None]) * cat[None]
        + time[:, None, None, None, None] * dog[None]
    )
    decoded_path = torch.cat(
        [
            decode_batches(
                vae, frame, batch_size=args.vae_batch_size, device=device
            )[None]
            for frame in path
        ]
    ).transpose(0, 1)
    decoded_cat = decode_batches(
        vae, cat, batch_size=args.vae_batch_size, device=device
    )
    decoded_dog = decode_batches(
        vae, dog, batch_size=args.vae_batch_size, device=device
    )
    make_domain_contact_sheet(
        pil_images(decoded_cat),
        pil_paths(decoded_path),
        pil_images(decoded_dog),
        args.output_dir / f"{args.cost}_ot_exact_kl_chords.png",
        source_name="paired cat",
        target_name="paired dog",
        clock_label="exact frozen KL-VAE chord; upper bound for pairwise overfit",
    )
    distance = (cat - dog).square().flatten(1).mean(1).cpu()
    summary = {
        "cost": args.cost,
        "pair_indices": indices.tolist(),
        "pair_latent_mse": distance.tolist(),
        "pair_latent_mse_mean": float(distance.mean()),
        "ot_spatial_weight": args.ot_spatial_weight,
        "ot_spatial_pool": args.ot_spatial_pool,
    }
    (args.output_dir / f"{args.cost}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument("--ot-feature-cache", type=Path, default=project / "outputs/distributional_latent_transport_full/train_dino_content_features.pt")
    parser.add_argument("--spatial-ot-feature-cache", type=Path, default=project / "outputs/bidirectional_kl_vae_anchored_sde/train_dino_spatial_content_features.pt")
    parser.add_argument("--latent-cache", type=Path, default=project / "outputs/bidirectional_kl_vae_diffusion/train_kl_vae_latents.pt")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/kl_vae_ot_pair_interpolations")
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cost", choices=("global", "spatial"), default="spatial")
    parser.add_argument("--ot-spatial-weight", type=float, default=0.2)
    parser.add_argument("--ot-spatial-pool", type=int, default=2)
    parser.add_argument("--ot-batch-size", type=int, default=64)
    parser.add_argument("--ot-epsilon", type=float, default=0.03)
    parser.add_argument("--ot-iterations", type=int, default=75)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--display-steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
