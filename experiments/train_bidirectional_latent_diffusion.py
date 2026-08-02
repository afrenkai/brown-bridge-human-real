"""Train one stochastic endpoint kernel for unpaired cat↔dog latent transport."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.latent_diffusion import (
    BidirectionalLatentDenoiser,
    conditional_latent_ddim_sample,
    q_sample,
)
from brown_bridge.light_sb import expected_minibatch_ot_pairs
from brown_bridge.residual_diffusion import cosine_alpha_bar
from render_dino_stargan_bridge import dino_encode, tensor_to_pil
from render_full_dataset_demo import fit_or_load_afhq_probe, load_models
from train_continuous_vae_renderer import AFHQDomainDataset
from train_distributional_latent_transport import (
    endpoint_distribution_metrics,
    make_domain_contact_sheet,
    path_metrics,
    pil_paths,
)


def load_training_pairs(
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    latent = torch.load(args.coordinate_cache, map_location="cpu", weights_only=False)
    semantic = torch.load(args.ot_feature_cache, map_location="cpu", weights_only=False)
    if latent.get("names") != semantic.get("names") or latent.get("factor") != 1:
        raise RuntimeError("latent and semantic caches do not describe the same full dataset")
    # The autoencoder ends in tanh, so raw latents have a shared compact
    # support. Per-coordinate standardization breaks translation equivariance
    # and lets a convolutional sampler leave the decoder's valid support.
    cat = latent["cat"].float()
    dog = latent["dog"].float()
    pair_count = min(len(cat), len(dog)) // args.ot_batch_size
    paired_cat, paired_dog = expected_minibatch_ot_pairs(
        cat,
        dog,
        batch_size=args.ot_batch_size,
        batches=pair_count,
        epsilon=args.ot_epsilon,
        sinkhorn_iterations=args.ot_iterations,
        generator=torch.Generator().manual_seed(args.seed + 1),
        source_cost_features=semantic["cat_content"],
        target_cost_features=semantic["dog_content"],
        pairing="rounded",
    )
    return paired_cat, paired_dog, {"cat": len(cat), "dog": len(dog), "pairs": len(paired_cat)}


def train_endpoint_kernel(
    cat: Tensor,
    dog: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BidirectionalLatentDenoiser, list[dict[str, float]]]:
    model = BidirectionalLatentDenoiser(
        args.latent_channels,
        base_channels=args.model_width,
        time_dim=args.time_dim,
        latent_size=args.latent_size,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    alpha_bar = cosine_alpha_bar(args.diffusion_steps, device=device)
    generator = torch.Generator().manual_seed(args.seed + 2)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, args.steps + 1):
        index = torch.randint(len(cat), (args.batch_size,), generator=generator)
        cat_batch = cat[index].reshape(
            -1, args.latent_channels, args.latent_size, args.latent_size
        ).to(device)
        dog_batch = dog[index].reshape_as(cat_batch).to(device)
        direction = 2 * torch.randint(
            0, 2, (args.batch_size,), generator=generator
        ) - 1
        direction = direction.to(device)
        forward = direction > 0
        select = forward[:, None, None, None]
        condition = torch.where(select, cat_batch, dog_batch)
        endpoint = torch.where(select, dog_batch, cat_batch)
        if args.prediction_target == "residual":
            clean = endpoint - condition
        elif args.prediction_target == "domain":
            clean = endpoint
            condition = torch.zeros_like(condition)
        else:
            clean = endpoint
        if args.condition_dropout > 0:
            dropped = torch.rand(args.batch_size, generator=generator) < args.condition_dropout
            condition = condition.masked_fill(dropped.to(device)[:, None, None, None], 0.0)
        time_index = torch.randint(
            args.diffusion_steps, (args.batch_size,), generator=generator
        ).to(device)
        noise = torch.randn(clean.shape, generator=generator).to(device)
        noisy = q_sample(clean, noise, time_index, alpha_bar)
        time = time_index.float() / max(args.diffusion_steps - 1, 1)
        prediction = model(noisy, condition, time, direction)
        per_sample = (prediction - noise).square().mean((1, 2, 3))
        cumulative = alpha_bar[time_index]
        snr = cumulative / (1.0 - cumulative).clamp_min(1e-6)
        min_snr_weight = torch.minimum(snr, torch.full_like(snr, args.min_snr)) / snr
        shape = (len(clean),) + (1,) * (clean.ndim - 1)
        predicted_clean = (
            noisy - (1.0 - cumulative).sqrt().reshape(shape) * prediction
        ) / cumulative.sqrt().reshape(shape).clamp_min(1e-4)
        x0 = F.smooth_l1_loss(predicted_clean, clean, reduction="none").mean((1, 2, 3))
        low_frequency = F.smooth_l1_loss(
            F.avg_pool2d(predicted_clean, 4),
            F.avg_pool2d(clean, 4),
            reduction="none",
        ).mean((1, 2, 3))
        loss = (
            min_snr_weight
            * (
                per_sample
                + args.lambda_x0 * x0
                + args.lambda_low_frequency * low_frequency
            )
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": float(step),
                "weighted_noise_mse": float(loss.detach()),
                "noise_mse": float(per_sample.mean().detach()),
                "x0_smooth_l1": float(x0.mean().detach()),
                "low_frequency_smooth_l1": float(low_frequency.mean().detach()),
            }
            history.append(record)
            print("latent diffusion " + json.dumps(record), flush=True)
    return model.eval(), history


@torch.inference_mode()
def decode_raw_path(
    autoencoder: torch.nn.Module, path: Tensor, batch_size: int
) -> Tensor:
    """Decode a path shaped [time,batch,channels,height,width]."""
    time_count, batch = path.shape[:2]
    decoded = [autoencoder.decode(value) for value in path.flatten(0, 1).split(batch_size)]
    return torch.cat(decoded).reshape(time_count, batch, 3, 128, 128).transpose(0, 1).cpu()


@torch.inference_mode()
def encode_images(model: torch.nn.Module, images: Tensor, batch_size: int, device: torch.device) -> Tensor:
    return torch.cat([model.encode(batch.to(device)) for batch in images.split(batch_size)])


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    device = torch.device("mps")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    autoencoder, dino, prior_summary = load_models(args, device)
    preprocess = BatchPreprocessor(64)
    probe, species_direction, probe_test = fit_or_load_afhq_probe(dino, args, device)
    paired_cat, paired_dog, train_counts = load_training_pairs(args)
    if args.evaluate_only:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
        model, history = train_endpoint_kernel(paired_cat, paired_dog, args, device)
        torch.save({"model": model.state_dict(), "history": history}, args.checkpoint)

    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=False, augment=False
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
    latents = {
        domain: encode_images(autoencoder, value, args.decode_batch_size, device)
        for domain, value in images.items()
    }
    coordinates = latents
    alpha_bar = cosine_alpha_bar(args.diffusion_steps, device=device)
    noise_generator = torch.Generator().manual_seed(args.seed + 4)
    forward_noise = torch.randn(coordinates["cat"].shape, generator=noise_generator).to(device)
    backward_noise = torch.randn(coordinates["dog"].shape, generator=noise_generator).to(device)
    generated_dog = conditional_latent_ddim_sample(
        model,
        coordinates["cat"],
        alpha_bar,
        direction=1,
        sampling_steps=args.sampling_steps,
        initial_noise=forward_noise,
        clean_clip=1.0,
        start_fraction=args.sampling_start,
        guidance_scale=args.guidance_scale,
    )
    generated_cat = conditional_latent_ddim_sample(
        model,
        coordinates["dog"],
        alpha_bar,
        direction=-1,
        sampling_steps=args.sampling_steps,
        initial_noise=backward_noise,
        clean_clip=1.0,
        start_fraction=args.sampling_start,
        guidance_scale=args.guidance_scale,
    )
    time = torch.linspace(0, 1, args.display_steps, device=device)
    progress = 0.5 - 0.5 * torch.cos(torch.pi * time)
    forward_path = torch.lerp(
        coordinates["cat"][: args.examples, None],
        generated_dog[: args.examples, None],
        progress[None, :, None, None, None],
    ).transpose(0, 1)
    backward_path = torch.lerp(
        coordinates["dog"][: args.examples, None],
        generated_cat[: args.examples, None],
        progress[None, :, None, None, None],
    ).transpose(0, 1)
    forward_decoded = decode_raw_path(autoencoder, forward_path, args.decode_batch_size)
    backward_decoded = decode_raw_path(autoencoder, backward_path, args.decode_batch_size)
    endpoint_dog = torch.cat(
        [autoencoder.decode(value) for value in generated_dog.split(args.decode_batch_size)]
    ).cpu()
    endpoint_cat = torch.cat(
        [autoencoder.decode(value) for value in generated_cat.split(args.decode_batch_size)]
    ).cpu()

    forward_metrics, _ = path_metrics(
        forward_decoded, dino, probe, preprocess, device, direction=1
    )
    backward_metrics, _ = path_metrics(
        backward_decoded, dino, probe, preprocess, device, direction=-1
    )
    real_features = {
        domain: dino_encode(dino, value, preprocess, device)[0]
        for domain, value in images.items()
    }
    endpoint_features = {
        "dog": dino_encode(dino, endpoint_dog, preprocess, device)[0],
        "cat": dino_encode(dino, endpoint_cat, preprocess, device)[0],
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

    def pil_images(batch: Tensor) -> list[Image.Image]:
        return [
            tensor_to_pil(value.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
            for value in batch
        ]

    sheet_index = torch.linspace(0, args.display_steps - 1, args.sheet_steps).round().long().tolist()
    make_domain_contact_sheet(
        pil_images(images["cat"][: args.examples]),
        [[row[index] for index in sheet_index] for row in pil_paths(forward_decoded)],
        pil_images(images["dog"][: args.examples]),
        args.output_dir / "latent_diffusion_cat_to_dog.png",
        source_name="cat", target_name="dog", clock_label="conditional latent diffusion + bridge",
    )
    make_domain_contact_sheet(
        pil_images(images["dog"][: args.examples]),
        [[row[index] for index in sheet_index] for row in pil_paths(backward_decoded)],
        pil_images(images["cat"][: args.examples]),
        args.output_dir / "latent_diffusion_dog_to_cat.png",
        source_name="dog", target_name="cat", clock_label="same model, reverse direction condition",
    )
    summary: dict[str, object] = {
        "method": "direction-conditioned latent diffusion endpoint kernel with endpoint-pinned bridge",
        "train_counts": train_counts,
        "test_counts": {key: len(value) for key, value in datasets.items()},
        "metrics": {"cat_to_dog": forward_metrics, "dog_to_cat": backward_metrics},
        "probe_test": probe_test,
        "history": history,
        "quarantine_verified": prior_summary["quarantine_verified"],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {"cat_to_dog": forward_decoded, "dog_to_cat": backward_decoded},
        args.output_dir / "trajectories.pt",
    )
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument("--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt")
    parser.add_argument("--model-checkpoint", type=Path, default=project / "outputs/full_dataset_latent_bridge/full_dataset_latent_bridge.pt")
    parser.add_argument("--transport-checkpoint", type=Path, default=project / "outputs/distributional_latent_transport_full/trained_transport_checkpoint.pt")
    parser.add_argument("--coordinate-cache", type=Path, default=project / "outputs/distributional_latent_transport_full/train_coarse_latents.pt")
    parser.add_argument("--ot-feature-cache", type=Path, default=project / "outputs/distributional_latent_transport_full/train_dino_content_features.pt")
    parser.add_argument("--probe-cache", type=Path, default=project / "outputs/full_dataset_latent_bridge_demo/afhq_dino_linear_probe.pt")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/bidirectional_latent_diffusion")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=24)
    parser.add_argument("--sampling-start", type=float, default=1.0)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--lambda-x0", type=float, default=0.05)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.1)
    parser.add_argument(
        "--prediction-target",
        choices=("endpoint", "residual", "domain"),
        default="endpoint",
    )
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--model-width", type=int, default=48)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--transformer-blocks", type=int, default=0)
    parser.add_argument("--transformer-heads", type=int, default=6)
    parser.add_argument("--ot-batch-size", type=int, default=64)
    parser.add_argument("--ot-epsilon", type=float, default=0.03)
    parser.add_argument("--ot-iterations", type=int, default=75)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "latent_diffusion.pt"
    if not 2 <= args.sheet_steps <= args.display_steps:
        parser.error("invalid display step counts")
    if not 1 <= args.examples <= args.eval_examples:
        parser.error("examples must not exceed eval-examples")
    return args


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(run(configuration), indent=2))
