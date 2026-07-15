"""Render exact endpoint-conditioned components of the learned latent SB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.features import content_cost_matrix, content_descriptor, species_direction
from brown_bridge.latent_image_bridge import LatentImageAutoencoder
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.paths import CoupledEndpoints, brownian_bridge, make_time_grid
from render_dino_stargan_bridge import (
    DISPLAY_STEPS,
    dino_encode,
    image_batch,
    load_square,
    make_contact_sheet,
    make_distribution_gif,
    tensor_to_pil,
    unique_minimum_assignment,
)
from train_continuous_vae_renderer import image_edges
from train_latent_schrodinger_bridge import make_two_way_comparison


def make_baseline_comparison(baseline: Path, current: Path, output: Path) -> None:
    with Image.open(baseline) as image:
        top = image.convert("RGB")
    with Image.open(current) as image:
        bottom = image.convert("RGB")
    width, header = 1600, 42
    top = top.resize((width, round(width * top.height / top.width)), Image.Resampling.LANCZOS)
    bottom = bottom.resize(
        (width, round(width * bottom.height / bottom.width)), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (width, top.height + bottom.height + 2 * header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((9, 14), "8x8 beta-VAE bridge (previous accepted renderer)", fill="black")
    canvas.paste(top, (0, header))
    second = header + top.height
    draw.text((9, second + 14), "32x32 endpoint-conditioned latent Schrodinger bridge", fill="black")
    canvas.paste(bottom, (0, second + header))
    canvas.save(output)


def load_autoencoder(args: argparse.Namespace, device: torch.device) -> LatentImageAutoencoder:
    payload = torch.load(args.autoencoder_checkpoint, map_location="cpu", weights_only=False)
    model = LatentImageAutoencoder(args.latent_channels, args.base_channels)
    model.load_state_dict(payload["model"])
    return model.eval().requires_grad_(False).to(device)


def load_dino(args: argparse.Namespace, device: torch.device) -> DinoCatDogClassifier:
    model = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb", image_size=64, pretrained=False, fusion="gated"
    )
    model.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    device = torch.device("mps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    autoencoder = load_autoencoder(args, device)
    dino = load_dino(args, device)
    source_paths = sorted((args.assets / "src/cat").glob("*.jpg"))[:5]
    reference_paths = sorted((args.assets / "ref/dog").glob("*.jpg"))
    sources = [load_square(path) for path in source_paths]
    dogs = [load_square(path) for path in reference_paths]
    source_tensor = image_batch(sources)
    dog_tensor = image_batch(dogs)
    preprocess = BatchPreprocessor(64)
    cat_global, cat_patch, _ = dino_encode(dino, source_tensor, preprocess, device)
    dog_global, dog_patch, _ = dino_encode(dino, dog_tensor, preprocess, device)
    direction = species_direction(cat_global, dog_global)
    cost = content_cost_matrix(
        content_descriptor(cat_global, cat_patch, direction),
        content_descriptor(dog_global, dog_patch, direction),
    )
    assignment = unique_minimum_assignment(cost)
    paired_dog_tensor = dog_tensor[assignment]
    paired_dogs = [dogs[index] for index in assignment]
    cats_128 = F.interpolate(source_tensor, size=(128, 128), mode="bilinear", align_corners=False)
    dogs_128 = F.interpolate(paired_dog_tensor, size=(128, 128), mode="bilinear", align_corners=False)
    cat_latent = autoencoder.encode(cats_128.to(device)).cpu()
    dog_latent = autoencoder.encode(dogs_128.to(device)).cpu()
    times = make_time_grid(DISPLAY_STEPS - 1, args.microsteps_per_macro)
    endpoints = CoupledEndpoints(
        cat_latent.flatten(1),
        dog_latent.flatten(1),
        torch.arange(len(cat_latent)),
        torch.arange(len(dog_latent)),
    )
    mean_flat = brownian_bridge(endpoints, times, diffusivity=0.0)
    stochastic_flat = brownian_bridge(
        endpoints,
        times,
        diffusivity=args.diffusivity,
        generator=torch.Generator().manual_seed(args.seed),
    )
    display_index = torch.arange(0, len(times), args.microsteps_per_macro)

    def decode(flat: Tensor) -> tuple[Tensor, list[list]]:
        selected = flat[display_index].reshape(
            DISPLAY_STEPS, len(sources), args.latent_channels, 32, 32
        )
        decoded = autoencoder.decode(selected.flatten(0, 1).float().to(device)).reshape(
            DISPLAY_STEPS, len(sources), 3, 128, 128
        ).transpose(0, 1)
        decoded[:, 0] = cats_128.to(device)
        decoded[:, -1] = dogs_128.to(device)
        pil = [
            [
                tensor_to_pil(image.mul(2).sub(1)).resize((256, 256))
                for image in row
            ]
            for row in decoded.cpu()
        ]
        return decoded, pil

    mean_images, mean_pil = decode(mean_flat)
    stochastic_images, stochastic_pil = decode(stochastic_flat)
    make_contact_sheet(
        sources,
        mean_pil,
        paired_dogs,
        args.output_dir / "conditional_latent_bridge.png",
        clock_label="conditional SB",
    )
    make_distribution_gif(
        mean_pil,
        args.output_dir / "conditional_latent_bridge.gif",
        clock_label="conditional SB",
    )
    make_two_way_comparison(
        mean_pil,
        stochastic_pil,
        args.output_dir / "conditional_mean_vs_brownian.png",
        deterministic_label="conditional mean",
        stochastic_label="Brownian bridge",
    )
    if args.baseline_sheet.exists():
        make_baseline_comparison(
            args.baseline_sheet,
            args.output_dir / "conditional_latent_bridge.png",
            args.output_dir / "vae_vs_conditional_latent_bridge.png",
        )
    all_images = torch.cat([mean_images.flatten(0, 1), stochastic_images.flatten(0, 1)])
    features, _, probabilities = dino_encode(dino, all_images.cpu(), preprocess, device)
    features = features.reshape(2, len(sources), DISPLAY_STEPS, -1)
    probabilities = probabilities.reshape(2, len(sources), DISPLAY_STEPS)
    coordinates = features @ direction.to(features)
    mean_dx, mean_dy = image_edges(mean_images[:, 1:-1].flatten(0, 1))
    stochastic_dx, stochastic_dy = image_edges(stochastic_images[:, 1:-1].flatten(0, 1))
    summary: dict[str, object] = {
        "method": "endpoint-conditioned components of the empirical latent Schrodinger bridge",
        "latent_shape": [args.latent_channels, 32, 32],
        "macro_steps": DISPLAY_STEPS - 1,
        "microsteps_per_macro": args.microsteps_per_macro,
        "diffusivity": args.diffusivity,
        "paired_dog_indices": assignment,
        "conditional_mean_dog_probability": probabilities[0].tolist(),
        "brownian_dog_probability": probabilities[1].tolist(),
        "conditional_mean_species_coordinate": coordinates[0].tolist(),
        "brownian_species_coordinate": coordinates[1].tolist(),
        "conditional_mean_interior_edge_energy": float(
            0.5 * (mean_dx.abs().mean() + mean_dy.abs().mean())
        ),
        "brownian_interior_edge_energy": float(
            0.5 * (stochastic_dx.abs().mean() + stochastic_dy.abs().mean())
        ),
        "conditional_mean_adjacent_l1": float(
            (mean_images[:, 1:] - mean_images[:, :-1]).abs().mean()
        ),
        "brownian_adjacent_l1": float(
            (stochastic_images[:, 1:] - stochastic_images[:, :-1]).abs().mean()
        ),
    }
    torch.save(
        {
            "conditional_mean_128": mean_images.cpu(),
            "brownian_128": stochastic_images.cpu(),
            "times": times,
            "conditional_mean_latent": mean_flat,
            "brownian_latent": stochastic_flat,
        },
        args.output_dir / "conditional_latent_trajectories.pt",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=project / "outputs/latent_schrodinger_bridge/latent_autoencoder.pt",
    )
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt"
    )
    parser.add_argument(
        "--assets", type=Path, default=project / "data/stargan-v2/assets/representative/afhq"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-sheet",
        type=Path,
        default=project / "outputs/manifold_vae_renderer_final/continuous_vae_contact_sheet.png",
    )
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--microsteps-per-macro", type=int, default=4)
    parser.add_argument("--diffusivity", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
