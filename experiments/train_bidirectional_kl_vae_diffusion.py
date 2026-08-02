"""Bidirectional AFHQ transport on a pretrained KL-VAE latent manifold.

The earlier deterministic reconstruction autoencoder accepts latents that are
easy for a denoiser to produce but invalid for its decoder.  This experiment
instead uses the frozen Stable Diffusion KL-VAE, trains one direction-
conditioned stochastic endpoint kernel, and decodes the complete latent path.
No target image or target DINO feature is supplied at held-out inference time.
"""

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
from brown_bridge.latent_diffusion import conditional_latent_ddim_sample
from brown_bridge.light_sb import expected_minibatch_ot_pairs
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.residual_diffusion import cosine_alpha_bar
from render_dino_stargan_bridge import dino_encode, tensor_to_pil
from render_full_dataset_demo import fit_or_load_afhq_probe
from train_bidirectional_latent_diffusion import train_endpoint_kernel
from train_continuous_vae_renderer import AFHQDomainDataset
from train_distributional_latent_transport import (
    endpoint_distribution_metrics,
    make_domain_contact_sheet,
    path_metrics,
    pil_paths,
)


class FrozenKLAutoencoder:
    """Small adapter around diffusers.AutoencoderKL with its native scaling."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval().requires_grad_(False)
        self.scaling_factor = float(model.config.scaling_factor)

    @torch.inference_mode()
    def encode(self, images: Tensor) -> Tensor:
        posterior = self.model.encode(images.mul(2).sub(1)).latent_dist
        return posterior.mode() * self.scaling_factor

    @torch.inference_mode()
    def decode(self, latents: Tensor) -> Tensor:
        decoded = self.model.decode(latents / self.scaling_factor).sample
        return decoded.add(1).div(2).clamp(0, 1)


def load_vae(args: argparse.Namespace, device: torch.device) -> FrozenKLAutoencoder:
    from diffusers import AutoencoderKL

    model = AutoencoderKL.from_pretrained(
        args.vae_model,
        local_files_only=args.local_files_only,
    ).to(device)
    return FrozenKLAutoencoder(model)


def load_dino(args: argparse.Namespace, device: torch.device) -> DinoCatDogClassifier:
    model = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb",
        image_size=64,
        pretrained=False,
        fusion="gated",
    )
    model.load_state_dict(
        torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True)
    )
    return model.eval().requires_grad_(False).to(device)


def encode_batches(
    vae: FrozenKLAutoencoder,
    images: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    return torch.cat(
        [vae.encode(batch.to(device)).cpu() for batch in images.split(batch_size)]
    )


def decode_batches(
    vae: FrozenKLAutoencoder,
    latents: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    return torch.cat(
        [vae.decode(batch.to(device)).cpu() for batch in latents.split(batch_size)]
    )


def _ordered_training_images(
    root: Path, domain: str, names: list[str]
) -> tuple[AFHQDomainDataset, list[int]]:
    dataset = AFHQDomainDataset(
        root, domain, image_size=128, training=True, augment=False
    )
    by_name = {path.name: index for index, path in enumerate(dataset.paths)}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"{len(missing)} cached DINO names are missing from AFHQ")
    return dataset, [by_name[name] for name in names]


def latent_cache(
    vae: FrozenKLAutoencoder,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, list[str]]]:
    semantic = torch.load(args.ot_feature_cache, map_location="cpu", weights_only=False)
    names = semantic["names"]
    if args.latent_cache.exists():
        cached = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
        if cached.get("names") != names or cached.get("vae_model") != args.vae_model:
            raise RuntimeError("KL-VAE cache metadata does not match this experiment")
        return (
            {domain: cached[domain].float() for domain in ("cat", "dog")},
            {domain: semantic[f"{domain}_content"].float() for domain in ("cat", "dog")},
            names,
        )

    latents: dict[str, Tensor] = {}
    for domain in ("cat", "dog"):
        dataset, ordered_indices = _ordered_training_images(
            args.afhq_root, domain, names[domain]
        )
        batches: list[Tensor] = []
        for start in range(0, len(ordered_indices), args.vae_batch_size):
            batch = torch.stack(
                [
                    dataset[index]
                    for index in ordered_indices[start : start + args.vae_batch_size]
                ]
            )
            batches.append(vae.encode(batch.to(device)).cpu())
            index = start // args.vae_batch_size + 1
            if index % 25 == 0:
                print(f"encoded {domain} batches={index}", flush=True)
        latents[domain] = torch.cat(batches)
    args.latent_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cat": latents["cat"].half(),
            "dog": latents["dog"].half(),
            "names": names,
            "vae_model": args.vae_model,
            "scaling_factor": vae.scaling_factor,
        },
        args.latent_cache,
    )
    return (
        latents,
        {domain: semantic[f"{domain}_content"].float() for domain in ("cat", "dog")},
        names,
    )


def pair_latents(
    latents: dict[str, Tensor],
    content: dict[str, Tensor],
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor]:
    pair_batches = min(len(latents["cat"]), len(latents["dog"])) // args.ot_batch_size
    shape = latents["cat"].shape[1:]
    cost_features = content
    spatial_weight = float(getattr(args, "ot_spatial_weight", 0.0))
    if not 0.0 <= spatial_weight < 1.0:
        raise ValueError("ot_spatial_weight must lie in [0, 1)")
    if spatial_weight:
        pool = int(getattr(args, "ot_spatial_pool", 4))
        if pool < 1 or shape[-1] % pool:
            raise ValueError("ot_spatial_pool must divide the latent spatial size")
        spatial = {
            domain: F.avg_pool2d(latents[domain], pool).flatten(1)
            for domain in ("cat", "dog")
        }
        joined = torch.cat([spatial["cat"], spatial["dog"]])
        mean = joined.mean(0)
        scale = joined.std(0, unbiased=False).clamp_min(1e-4)
        cost_features = {
            domain: torch.cat(
                [
                    (1.0 - spatial_weight) ** 0.5
                    * F.normalize(content[domain], dim=1),
                    spatial_weight**0.5
                    * F.normalize((spatial[domain] - mean) / scale, dim=1),
                ],
                dim=1,
            )
            for domain in ("cat", "dog")
        }
    paired_cat, paired_dog = expected_minibatch_ot_pairs(
        latents["cat"].flatten(1),
        latents["dog"].flatten(1),
        batch_size=args.ot_batch_size,
        batches=pair_batches,
        epsilon=args.ot_epsilon,
        sinkhorn_iterations=args.ot_iterations,
        generator=torch.Generator().manual_seed(args.seed + 1),
        source_cost_features=cost_features["cat"],
        target_cost_features=cost_features["dog"],
        pairing="rounded",
    )
    return paired_cat.reshape(-1, *shape), paired_dog.reshape(-1, *shape)


def pil_images(images: Tensor) -> list[Image.Image]:
    return [
        tensor_to_pil(image.mul(2).sub(1)).resize(
            (256, 256), Image.Resampling.LANCZOS
        )
        for image in images
    ]


def reconstruction_sheet(
    originals: Tensor, reconstructions: Tensor, output: Path
) -> None:
    tile, header = 256, 36
    canvas = Image.new("RGB", (2 * tile, header + len(originals) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "held-out original", fill="black")
    draw.text((tile + 8, 8), "frozen KL-VAE reconstruction", fill="black")
    for row, (original, reconstruction) in enumerate(
        zip(pil_images(originals), pil_images(reconstructions))
    ):
        y = header + row * tile
        canvas.paste(original, (0, y))
        canvas.paste(reconstruction, (tile, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.inference_mode()
def decode_path(
    vae: FrozenKLAutoencoder,
    path: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    time_count, batch = path.shape[:2]
    decoded = decode_batches(
        vae, path.flatten(0, 1), batch_size=batch_size, device=device
    )
    return decoded.reshape(time_count, batch, 3, 128, 128).transpose(0, 1)


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

    latents, content, names = latent_cache(vae, args, device)
    paired_cat, paired_dog = pair_latents(latents, content, args)
    args.latent_channels = paired_cat.shape[1]
    args.latent_size = paired_cat.shape[-1]
    if args.evaluate_only:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        from brown_bridge.latent_diffusion import BidirectionalLatentDenoiser

        model = BidirectionalLatentDenoiser(
            args.latent_channels,
            base_channels=args.model_width,
            time_dim=args.time_dim,
            latent_size=args.latent_size,
            transformer_blocks=args.transformer_blocks,
            transformer_heads=args.transformer_heads,
        )
        model.load_state_dict(saved["model"])
        model = model.eval().to(device)
        history = saved["history"]
    else:
        model, history = train_endpoint_kernel(
            paired_cat, paired_dog, args, device
        )
        torch.save({"model": model.state_dict(), "history": history}, args.checkpoint)

    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root,
            domain,
            image_size=128,
            training=False,
            augment=False,
        )
        for domain in ("cat", "dog")
    }
    generator = torch.Generator().manual_seed(args.seed + 3)
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
    reconstructed = decode_batches(
        vae,
        torch.cat([encoded["cat"][:3], encoded["dog"][:3]]),
        batch_size=args.vae_batch_size,
        device=device,
    )
    reconstruction_original = torch.cat([images["cat"][:3], images["dog"][:3]])
    reconstruction_sheet(
        reconstruction_original,
        reconstructed,
        args.output_dir / "kl_vae_reconstructions.png",
    )
    reconstruction_l1 = float(F.l1_loss(reconstructed, reconstruction_original))

    alpha_bar = cosine_alpha_bar(args.diffusion_steps, device=device)
    noise_generator = torch.Generator().manual_seed(args.seed + 4)
    noises = {
        domain: torch.randn(value.shape, generator=noise_generator).to(device)
        for domain, value in encoded.items()
    }
    model_condition = (
        {domain: torch.zeros_like(value) for domain, value in encoded.items()}
        if args.prediction_target == "domain"
        else encoded
    )
    initial_clean = encoded if args.prediction_target == "domain" else {"cat": None, "dog": None}
    sampled = {
        "dog": conditional_latent_ddim_sample(
            model,
            model_condition["cat"],
            alpha_bar,
            direction=1,
            sampling_steps=args.sampling_steps,
            initial_noise=noises["cat"],
            clean_clip=args.clean_clip,
            guidance_scale=args.guidance_scale,
            start_fraction=args.sampling_start,
            initial_clean=initial_clean["cat"],
        ),
        "cat": conditional_latent_ddim_sample(
            model,
            model_condition["dog"],
            alpha_bar,
            direction=-1,
            sampling_steps=args.sampling_steps,
            initial_noise=noises["dog"],
            clean_clip=args.clean_clip,
            guidance_scale=args.guidance_scale,
            start_fraction=args.sampling_start,
            initial_clean=initial_clean["dog"],
        ),
    }
    generated = (
        {
            "dog": encoded["cat"] + sampled["dog"],
            "cat": encoded["dog"] + sampled["cat"],
        }
        if args.prediction_target == "residual"
        else sampled
    )
    time = torch.linspace(0, 1, args.display_steps, device=device)
    progress = 0.5 - 0.5 * torch.cos(torch.pi * time)
    paths = {
        "cat_to_dog": torch.lerp(
            encoded["cat"][: args.examples, None],
            generated["dog"][: args.examples, None],
            progress[None, :, None, None, None],
        ).transpose(0, 1),
        "dog_to_cat": torch.lerp(
            encoded["dog"][: args.examples, None],
            generated["cat"][: args.examples, None],
            progress[None, :, None, None, None],
        ).transpose(0, 1),
    }
    decoded_paths = {
        name: decode_path(
            vae, path, batch_size=args.vae_batch_size, device=device
        )
        for name, path in paths.items()
    }
    decoded_endpoints = {
        domain: decode_batches(
            vae, value, batch_size=args.vae_batch_size, device=device
        )
        for domain, value in generated.items()
    }

    forward_metrics, _ = path_metrics(
        decoded_paths["cat_to_dog"], dino, probe, preprocess, device, direction=1
    )
    backward_metrics, _ = path_metrics(
        decoded_paths["dog_to_cat"], dino, probe, preprocess, device, direction=-1
    )
    real_features = {
        domain: dino_encode(dino, value, preprocess, device)[0]
        for domain, value in images.items()
    }
    endpoint_features = {
        domain: dino_encode(dino, value, preprocess, device)[0]
        for domain, value in decoded_endpoints.items()
    }
    forward_metrics.update(
        endpoint_distribution_metrics(
            real_features["cat"],
            endpoint_features["dog"],
            real_features["dog"],
            species_direction,
            target_name="dogs",
        )
    )
    backward_metrics.update(
        endpoint_distribution_metrics(
            real_features["dog"],
            endpoint_features["cat"],
            real_features["cat"],
            species_direction,
            target_name="cats",
        )
    )

    sheet_indices = (
        torch.linspace(0, args.display_steps - 1, args.sheet_steps)
        .round()
        .long()
        .tolist()
    )
    make_domain_contact_sheet(
        pil_images(images["cat"][: args.examples]),
        [
            [row[index] for index in sheet_indices]
            for row in pil_paths(decoded_paths["cat_to_dog"])
        ],
        pil_images(images["dog"][: args.examples]),
        args.output_dir / "kl_vae_cat_to_dog.png",
        source_name="cat",
        target_name="dog",
        clock_label="one stochastic endpoint kernel; frozen KL-VAE latent interpolation",
    )
    make_domain_contact_sheet(
        pil_images(images["dog"][: args.examples]),
        [
            [row[index] for index in sheet_indices]
            for row in pil_paths(decoded_paths["dog_to_cat"])
        ],
        pil_images(images["cat"][: args.examples]),
        args.output_dir / "kl_vae_dog_to_cat.png",
        source_name="dog",
        target_name="cat",
        clock_label="same stochastic endpoint kernel; reverse direction condition",
    )

    summary: dict[str, object] = {
        "method": "direction-conditioned diffusion on a frozen generative KL-VAE latent",
        "inference_uses_target": False,
        "vae_model": args.vae_model,
        "vae_scaling_factor": vae.scaling_factor,
        "training_counts": {
            "cats": len(latents["cat"]),
            "dogs": len(latents["dog"]),
            "ot_pairs": len(paired_cat),
        },
        "heldout_counts": {domain: len(dataset) for domain, dataset in datasets.items()},
        "heldout_vae_reconstruction_l1": reconstruction_l1,
        "metrics": {"cat_to_dog": forward_metrics, "dog_to_cat": backward_metrics},
        "probe_test": probe_test,
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cache_name_counts": {domain: len(value) for domain, value in names.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/bidirectional_kl_vae_diffusion")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--latent-cache", type=Path)
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--sampling-start", type=float, default=1.0)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--lambda-x0", type=float, default=0.05)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.1)
    parser.add_argument(
        "--prediction-target",
        choices=("endpoint", "residual", "domain"),
        default="residual",
    )
    parser.add_argument("--clean-clip", type=float, default=4.0)
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--ot-batch-size", type=int, default=64)
    parser.add_argument("--ot-epsilon", type=float, default=0.03)
    parser.add_argument("--ot-iterations", type=int, default=75)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "endpoint_kernel.pt"
    if args.latent_cache is None:
        args.latent_cache = args.output_dir / "train_kl_vae_latents.pt"
    if not 0 <= args.condition_dropout < 1:
        parser.error("condition-dropout must lie in [0,1)")
    if not 2 <= args.sheet_steps <= args.display_steps:
        parser.error("invalid display step counts")
    return args


if __name__ == "__main__":
    run(parse_args())
