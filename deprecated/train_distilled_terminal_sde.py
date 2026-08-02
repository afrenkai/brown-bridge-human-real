"""Distill the empirical spatial-OT endpoint kernel into a terminal diffusion.

The student receives only a source KL latent, its species-projected spatial-DINO
descriptor, and a direction.  It samples a terminal latent without retrieving a
support image at inference.  An exact pinned Brownian SDE then supplies the
continuous bidirectional image path.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.latent_diffusion import (
    BidirectionalLatentDenoiser,
    conditional_latent_ddim_sample,
    q_sample,
)
from brown_bridge.residual_diffusion import cosine_alpha_bar
from brown_bridge.spatial_stochastic import sample_pinned_brownian_bridge
from brown_bridge.terminal_distillation import (
    sample_terminal_indices,
    topk_terminal_support,
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
)
from train_bidirectional_kl_vae_rectified_flow import make_path_gif
from train_bidirectional_kl_vae_stochastic_interpolant import compact_path_metrics
from train_continuous_vae_renderer import AFHQDomainDataset
from train_distributional_latent_transport import (
    endpoint_distribution_metrics,
    make_domain_contact_sheet,
    pil_paths,
)
from render_empirical_spatial_ot_sde import heldout_spatial_descriptor


def find_latent_row(values: Tensor, query: Tensor) -> int:
    error = (values.float() - query.float()).square().flatten(1).mean(1)
    index = int(error.argmin())
    if float(error[index]) > 1e-10:
        raise RuntimeError("could not identify the memorized endpoint in the cache")
    return index


def build_or_load_teacher_cache(
    latents: dict[str, Tensor],
    global_content: dict[str, Tensor],
    spatial_content: dict[str, Tensor],
    names: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, object]:
    expected = {
        "names": names,
        "neighbors": args.teacher_neighbors,
        "layout_weight": args.teacher_layout_weight,
        "layout_pool": args.teacher_layout_pool,
        "excluded_pair_index": args.excluded_overfit_pair_index,
    }
    if args.teacher_cache.exists():
        saved = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
        if all(saved.get(key) == value for key, value in expected.items()):
            return saved

    # Reconstruct the deliberate overfit pair under the original global-DINO
    # coupling, then remove both endpoint identities from source/support pools.
    original_args = copy.copy(args)
    original_args.ot_spatial_weight = 0.0
    paired_cat, paired_dog = pair_latents(latents, global_content, original_args)
    pair_index = args.excluded_overfit_pair_index
    if not 0 <= pair_index < len(paired_cat):
        raise ValueError("excluded-overfit-pair-index is invalid")
    excluded = {
        "cat": find_latent_row(latents["cat"], paired_cat[pair_index]),
        "dog": find_latent_row(latents["dog"], paired_dog[pair_index]),
    }

    raw_layout = {
        domain: F.avg_pool2d(
            latents[domain].float(), args.teacher_layout_pool
        ).flatten(1)
        for domain in ("cat", "dog")
    }
    joined = torch.cat([raw_layout["cat"], raw_layout["dog"]])
    layout_mean = joined.mean(0)
    layout_scale = joined.std(0, unbiased=False).clamp_min(1e-4)
    layout = {
        domain: F.normalize(
            (raw_layout[domain] - layout_mean) / layout_scale, dim=1
        )
        for domain in ("cat", "dog")
    }
    available = {
        domain: torch.tensor(
            [index for index in range(len(latents[domain])) if index != excluded[domain]]
        )
        for domain in ("cat", "dog")
    }
    cat_to_dog_local, cat_to_dog_cost = topk_terminal_support(
        spatial_content["cat"][available["cat"]],
        spatial_content["dog"][available["dog"]],
        layout["cat"][available["cat"]],
        layout["dog"][available["dog"]],
        neighbors=args.teacher_neighbors,
        layout_weight=args.teacher_layout_weight,
        chunk_size=args.teacher_chunk_size,
    )
    dog_to_cat_local, dog_to_cat_cost = topk_terminal_support(
        spatial_content["dog"][available["dog"]],
        spatial_content["cat"][available["cat"]],
        layout["dog"][available["dog"]],
        layout["cat"][available["cat"]],
        neighbors=args.teacher_neighbors,
        layout_weight=args.teacher_layout_weight,
        chunk_size=args.teacher_chunk_size,
    )
    saved: dict[str, object] = {
        **expected,
        "excluded_indices": excluded,
        "cat_source_indices": available["cat"],
        "dog_source_indices": available["dog"],
        "cat_to_dog_indices": available["dog"][cat_to_dog_local],
        "cat_to_dog_costs": cat_to_dog_cost,
        "dog_to_cat_indices": available["cat"][dog_to_cat_local],
        "dog_to_cat_costs": dog_to_cat_cost,
        "layout_mean": layout_mean,
        "layout_scale": layout_scale,
    }
    args.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.teacher_cache)
    return saved


def make_model(
    args: argparse.Namespace, descriptor_dim: int
) -> BidirectionalLatentDenoiser:
    return BidirectionalLatentDenoiser(
        4,
        base_channels=args.model_width,
        time_dim=args.time_dim,
        latent_size=16,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
        descriptor_dim=descriptor_dim,
    )


def diffusion_target(source: Tensor, endpoint: Tensor, mode: str) -> Tensor:
    """Return the random variable whose VP-SDE is learned by the student."""
    if mode == "terminal":
        return endpoint
    if mode == "residual":
        return endpoint - source
    raise ValueError(f"unknown target mode: {mode}")


def terminal_from_diffusion_sample(
    source: Tensor, sample: Tensor, mode: str
) -> Tensor:
    """Map a sampled terminal-SDE variable back to an image latent endpoint."""
    if mode == "terminal":
        return sample
    if mode == "residual":
        return source + sample
    raise ValueError(f"unknown target mode: {mode}")


def training_batch(
    latents: dict[str, Tensor],
    descriptors: dict[str, Tensor],
    teacher: dict[str, object],
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch = args.batch_size
    forward = torch.rand(batch, generator=generator) < 0.5
    cat_sources = teacher["cat_source_indices"]
    dog_sources = teacher["dog_source_indices"]
    assert isinstance(cat_sources, Tensor) and isinstance(dog_sources, Tensor)
    if args.overfit_sources:
        cat_sources = cat_sources[: args.overfit_sources]
        dog_sources = dog_sources[: args.overfit_sources]
    cat_position = torch.randint(len(cat_sources), (batch,), generator=generator)
    dog_position = torch.randint(len(dog_sources), (batch,), generator=generator)
    source = torch.empty((batch, 4, 16, 16))
    descriptor = torch.empty((batch, descriptors["cat"].shape[1]))
    endpoint = torch.empty_like(source)
    direction = torch.where(forward, 1.0, -1.0)

    if bool(forward.any()):
        positions = cat_position[forward]
        source_indices = cat_sources[positions]
        support_indices = teacher["cat_to_dog_indices"]
        support_costs = teacher["cat_to_dog_costs"]
        assert isinstance(support_indices, Tensor) and isinstance(support_costs, Tensor)
        targets = sample_terminal_indices(
            support_indices[positions],
            support_costs[positions],
            temperature=(0.0 if args.overfit_sources else args.teacher_temperature),
            generator=generator,
        )
        source[forward] = latents["cat"][source_indices]
        descriptor[forward] = descriptors["cat"][source_indices]
        endpoint[forward] = latents["dog"][targets]
    backward = ~forward
    if bool(backward.any()):
        positions = dog_position[backward]
        source_indices = dog_sources[positions]
        support_indices = teacher["dog_to_cat_indices"]
        support_costs = teacher["dog_to_cat_costs"]
        assert isinstance(support_indices, Tensor) and isinstance(support_costs, Tensor)
        targets = sample_terminal_indices(
            support_indices[positions],
            support_costs[positions],
            temperature=(0.0 if args.overfit_sources else args.teacher_temperature),
            generator=generator,
        )
        source[backward] = latents["dog"][source_indices]
        descriptor[backward] = descriptors["dog"][source_indices]
        endpoint[backward] = latents["cat"][targets]
    return (
        source.to(device),
        F.normalize(descriptor, dim=1).to(device),
        endpoint.to(device),
        direction.to(device),
    )


def train_terminal_model(
    latents: dict[str, Tensor],
    descriptors: dict[str, Tensor],
    teacher: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BidirectionalLatentDenoiser, list[dict[str, float]]]:
    model = make_model(args, descriptors["cat"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    alpha_bar = cosine_alpha_bar(args.diffusion_steps, device=device)
    generator = torch.Generator().manual_seed(args.seed + 90)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, args.steps + 1):
        condition, descriptor, endpoint, direction = training_batch(
            latents, descriptors, teacher, args, generator, device
        )
        clean = diffusion_target(condition, endpoint, args.target_mode)
        if args.condition_dropout:
            dropped = (
                torch.rand(args.batch_size, generator=generator)
                < args.condition_dropout
            ).to(device)
            condition = condition.masked_fill(dropped[:, None, None, None], 0.0)
            descriptor = descriptor.masked_fill(dropped[:, None], 0.0)
        time_index = torch.randint(
            args.diffusion_steps, (args.batch_size,), generator=generator
        ).to(device)
        noise = torch.randn(clean.shape, generator=generator).to(device)
        noisy = q_sample(clean, noise, time_index, alpha_bar)
        time = time_index.float() / max(args.diffusion_steps - 1, 1)
        prediction = model(
            noisy, condition, time, direction, descriptor=descriptor
        )
        cumulative = alpha_bar[time_index]
        shape = (len(clean), 1, 1, 1)
        predicted_clean = (
            noisy - (1.0 - cumulative).sqrt().reshape(shape) * prediction
        ) / cumulative.sqrt().reshape(shape).clamp_min(1e-4)
        noise_mse = (prediction - noise).square().mean((1, 2, 3))
        x0 = F.smooth_l1_loss(
            predicted_clean, clean, reduction="none"
        ).mean((1, 2, 3))
        low_frequency = F.smooth_l1_loss(
            F.avg_pool2d(predicted_clean, 4),
            F.avg_pool2d(clean, 4),
            reduction="none",
        ).mean((1, 2, 3))
        snr = cumulative / (1.0 - cumulative).clamp_min(1e-6)
        snr_weight = torch.minimum(snr, torch.full_like(snr, args.min_snr)) / snr
        loss = (
            snr_weight
            * (
                noise_mse
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
                "weighted_loss": float(loss.detach()),
                "noise_mse": float(noise_mse.mean().detach()),
                "x0_smooth_l1": float(x0.mean().detach()),
                "low_frequency_smooth_l1": float(low_frequency.mean().detach()),
            }
            history.append(record)
            print("terminal distillation " + json.dumps(record), flush=True)
    return model.eval(), history


def heldout_evaluation_data(
    vae: object,
    dino: torch.nn.Module,
    preprocess: BatchPreprocessor,
    spatial_species_direction: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    generator = torch.Generator().manual_seed(args.seed + 100)
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
    descriptor = {
        domain: F.normalize(
            heldout_spatial_descriptor(
                image, dino, preprocess, spatial_species_direction, device
            ),
            dim=1,
        ).to(device)
        for domain, image in images.items()
    }
    return images, encoded, descriptor


def overfit_evaluation_data(
    vae: object,
    latents: dict[str, Tensor],
    descriptors: dict[str, Tensor],
    teacher: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    encoded: dict[str, Tensor] = {}
    descriptor: dict[str, Tensor] = {}
    teacher_endpoint: dict[str, Tensor] = {}
    for domain, target_domain, support_name in (
        ("cat", "dog", "cat_to_dog_indices"),
        ("dog", "cat", "dog_to_cat_indices"),
    ):
        source_indices = teacher[f"{domain}_source_indices"]
        support = teacher[support_name]
        assert isinstance(source_indices, Tensor) and isinstance(support, Tensor)
        count = min(args.eval_examples, args.overfit_sources, len(source_indices))
        encoded[domain] = latents[domain][source_indices[:count]].to(device)
        descriptor[domain] = F.normalize(
            descriptors[domain][source_indices[:count]], dim=1
        ).to(device)
        teacher_endpoint[domain] = latents[target_domain][support[:count, 0]].to(device)
    images = {
        domain: decode_batches(
            vae, value, batch_size=args.vae_batch_size, device=device
        )
        for domain, value in encoded.items()
    }
    return images, encoded, descriptor, teacher_endpoint


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
    probe, species_probe_direction, probe_test = fit_or_load_afhq_probe(
        dino, args, device
    )
    latents, global_content, names = latent_cache(vae, args, device)
    spatial_saved = torch.load(
        args.spatial_ot_feature_cache, map_location="cpu", weights_only=False
    )
    if spatial_saved.get("names") != names:
        raise RuntimeError("spatial DINO and latent caches are not aligned")
    descriptors = {
        domain: spatial_saved[domain].float() for domain in ("cat", "dog")
    }
    teacher = build_or_load_teacher_cache(
        latents, global_content, descriptors, names, args
    )
    descriptor_dim = descriptors["cat"].shape[1]
    if args.evaluate_only:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = make_model(args, descriptor_dim)
        model.load_state_dict(saved["model"])
        model = model.eval().to(device)
        history = saved["history"]
    else:
        model, history = train_terminal_model(
            latents, descriptors, teacher, args, device
        )
        torch.save(
            {
                "model": model.state_dict(),
                "history": history,
                "descriptor_dim": descriptor_dim,
                "overfit_sources": args.overfit_sources,
            },
            args.checkpoint,
        )

    teacher_endpoint: dict[str, Tensor] | None = None
    if args.overfit_sources:
        images, encoded, source_descriptor, teacher_endpoint = overfit_evaluation_data(
            vae, latents, descriptors, teacher, args, device
        )
    else:
        images, encoded, source_descriptor = heldout_evaluation_data(
            vae,
            dino,
            preprocess,
            spatial_saved["species_direction"].float(),
            args,
            device,
        )

    alpha_bar = cosine_alpha_bar(args.diffusion_steps, device=device)
    noise_generator = torch.Generator().manual_seed(args.seed + 110)
    noises = {
        domain: torch.randn(value.shape, generator=noise_generator).to(device)
        for domain, value in encoded.items()
    }
    sampled = {
        "cat_to_dog": conditional_latent_ddim_sample(
            model,
            encoded["cat"],
            alpha_bar,
            direction=1,
            sampling_steps=args.sampling_steps,
            initial_noise=noises["cat"],
            clean_clip=args.clean_clip,
            guidance_scale=args.guidance_scale,
            descriptor=source_descriptor["cat"],
        ),
        "dog_to_cat": conditional_latent_ddim_sample(
            model,
            encoded["dog"],
            alpha_bar,
            direction=-1,
            sampling_steps=args.sampling_steps,
            initial_noise=noises["dog"],
            clean_clip=args.clean_clip,
            guidance_scale=args.guidance_scale,
            descriptor=source_descriptor["dog"],
        ),
    }
    source_latents = {
        "cat_to_dog": encoded["cat"],
        "dog_to_cat": encoded["dog"],
    }
    generated = {
        name: terminal_from_diffusion_sample(
            source_latents[name], value, args.target_mode
        )
        for name, value in sampled.items()
    }
    paths = {
        name: sample_pinned_brownian_bridge(
            source.flatten(1),
            generated[name].flatten(1),
            epsilon=args.bridge_epsilon,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
        ).reshape(args.display_steps, len(source), *source.shape[1:])
        for name, source in source_latents.items()
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
            vae, value, batch_size=args.vae_batch_size, device=device
        )
        for name, value in generated.items()
    }
    real_features = {
        domain: dino_encode(dino, value, preprocess, device)[0]
        for domain, value in images.items()
    }
    metrics: dict[str, dict[str, object]] = {}
    probabilities: dict[str, Tensor] = {}
    for name, source_domain, target_domain, direction in (
        ("cat_to_dog", "cat", "dog", 1),
        ("dog_to_cat", "dog", "cat", -1),
    ):
        result, probability = compact_path_metrics(
            decoded_display[name], dino, probe, preprocess, device, direction=direction
        )
        probabilities[name] = probability
        endpoint_features = dino_encode(
            dino, decoded_endpoints[name], preprocess, device
        )[0]
        endpoint_probability = probe.probabilities(endpoint_features)
        result.update(
            {
                "all_source_dog_probability_mean": float(
                    probe.probabilities(real_features[source_domain]).mean()
                ),
                "all_endpoint_dog_probability_mean": float(endpoint_probability.mean()),
                "all_endpoint_classified_as_target_fraction": float(
                    (
                        (endpoint_probability >= 0.5)
                        if direction > 0
                        else (endpoint_probability < 0.5)
                    ).float().mean()
                ),
            }
        )
        if not args.overfit_sources:
            target_dataset = AFHQDomainDataset(
                args.afhq_root,
                target_domain,
                image_size=128,
                training=False,
                augment=False,
            )
            target_images = torch.stack(
                [target_dataset[index] for index in range(min(len(images[source_domain]), len(target_dataset)))]
            )
            target_features = dino_encode(
                dino, target_images, preprocess, device
            )[0]
            result.update(
                endpoint_distribution_metrics(
                    real_features[source_domain],
                    endpoint_features,
                    target_features,
                    species_probe_direction,
                    target_name=f"{target_domain}s",
                )
            )
        else:
            assert teacher_endpoint is not None
            key = source_domain
            teacher_value = teacher_endpoint[key]
            result["teacher_terminal_latent_mse"] = float(
                F.mse_loss(generated[name], teacher_value)
            )
        metrics[name] = result

    sheet_indices = torch.linspace(
        0, args.display_steps - 1, args.sheet_steps
    ).round().long().tolist()
    for name, source_domain, target_domain in (
        ("cat_to_dog", "cat", "dog"),
        ("dog_to_cat", "dog", "cat"),
    ):
        path_images = pil_paths(decoded_display[name])
        if args.overfit_sources:
            assert teacher_endpoint is not None
            references = decode_batches(
                vae,
                teacher_endpoint[source_domain][: args.examples],
                batch_size=args.vae_batch_size,
                device=device,
            )
            reference_label = "empirical teacher terminal"
        else:
            opposite = images[target_domain]
            references = opposite[: args.examples]
            reference_label = f"unrelated held-out {target_domain} ref"
        make_domain_contact_sheet(
            pil_images(images[source_domain][: args.examples]),
            [[row[index] for index in sheet_indices] for row in path_images],
            pil_images(references),
            args.output_dir / f"distilled_terminal_sde_{name}.png",
            source_name=source_domain,
            target_name=target_domain,
            clock_label="distilled terminal diffusion + exact pinned Brownian SDE",
            reference_label=reference_label,
        )
        make_path_gif(
            path_images,
            args.output_dir / f"distilled_terminal_sde_{name}.gif",
            source_name=source_domain,
            target_name=target_domain,
        )

    summary = {
        "method": (
            "spatial-DINO empirical-kernel distillation into conditional "
            f"{args.target_mode} diffusion"
        ),
        "terminal_inference_retrieves_training_support": False,
        "source_input_only": True,
        "overfit_diagnostic": bool(args.overfit_sources),
        "excluded_overfit_pair_indices": teacher["excluded_indices"],
        "teacher_neighbors": args.teacher_neighbors,
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
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument("--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt")
    parser.add_argument("--ot-feature-cache", type=Path, default=project / "outputs/distributional_latent_transport_full/train_dino_content_features.pt")
    parser.add_argument("--spatial-ot-feature-cache", type=Path, default=project / "outputs/bidirectional_kl_vae_anchored_sde/train_dino_spatial_content_features.pt")
    parser.add_argument("--probe-cache", type=Path, default=project / "outputs/full_dataset_latent_bridge_demo/afhq_dino_linear_probe.pt")
    parser.add_argument("--latent-cache", type=Path, default=project / "outputs/bidirectional_kl_vae_diffusion/train_kl_vae_latents.pt")
    parser.add_argument("--teacher-cache", type=Path, default=project / "outputs/distilled_terminal_sde/terminal_teacher_topk.pt")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/distilled_terminal_sde")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.25)
    parser.add_argument("--clean-clip", type=float, default=4.0)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--lambda-x0", type=float, default=0.1)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.1)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--teacher-neighbors", type=int, default=8)
    parser.add_argument("--teacher-temperature", type=float, default=0.02)
    parser.add_argument(
        "--target-mode", choices=("terminal", "residual"), default="terminal"
    )
    parser.add_argument("--teacher-layout-weight", type=float, default=0.2)
    parser.add_argument("--teacher-layout-pool", type=int, default=2)
    parser.add_argument("--teacher-chunk-size", type=int, default=256)
    parser.add_argument("--excluded-overfit-pair-index", type=int, default=0)
    parser.add_argument("--ot-batch-size", type=int, default=64)
    parser.add_argument("--ot-epsilon", type=float, default=0.03)
    parser.add_argument("--ot-iterations", type=int, default=75)
    parser.add_argument("--bridge-epsilon", type=float, default=0.01)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--overfit-sources", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "terminal_diffusion.pt"
    if args.overfit_sources < 0:
        parser.error("overfit-sources must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
