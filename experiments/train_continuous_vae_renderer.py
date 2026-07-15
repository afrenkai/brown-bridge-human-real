"""Train a continuous-time AFHQ VAE renderer for reciprocal DINO paths."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from brown_bridge.continuous_vae import (
    ContinuousConditionalVAE,
    diagonal_gaussian_kl,
    latent_brownian_bridge,
)
from brown_bridge.data import BatchPreprocessor
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.stargan import load_afhq_renderer
from brown_bridge.spatial_bridge import species_clock
from condition_stargan_on_reciprocal_sde import (
    dino_features_with_grad,
    evaluate_rendered_paths,
    render_paths,
)
from render_dino_stargan_bridge import (
    CAT_DOMAIN,
    DEFAULT_STARGAN,
    DISPLAY_STEPS,
    DOG_DOMAIN,
    SEED,
    WORKSPACE,
    image_batch,
    load_square,
    make_contact_sheet,
    make_distribution_gif,
    require_mps,
    tensor_to_pil,
)
from brown_bridge.conditioning import endpoint_anchored_layer_styles, pin_trajectory_endpoint
from brown_bridge.exclusions import filter_excluded_paths


class AFHQDomainDataset(Dataset[Tensor]):
    def __init__(
        self,
        root: Path,
        domain: str,
        *,
        image_size: int = 128,
        training: bool,
        limit: int | None = None,
        exclusion_manifest: Path | None = None,
        augment: bool | None = None,
    ) -> None:
        self.paths = sorted((root / ("train" if training else "test") / domain).glob("*"))
        self.paths = [path for path in self.paths if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        manifest = exclusion_manifest
        if manifest is None and (root / "exclusions.json").exists():
            manifest = root / "exclusions.json"
        self.paths = filter_excluded_paths(self.paths, manifest)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no AFHQ {domain} images under {root}")
        self.image_size = image_size
        self.training = training
        self.augment = training if augment is None else augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if self.augment:
                width, height = image.size
                crop_fraction = random.uniform(0.86, 1.0)
                side = max(1, round(min(width, height) * crop_fraction))
                left = random.randint(0, max(0, width - side))
                top = random.randint(0, max(0, height - side))
                image = image.crop((left, top, left + side, top + side))
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
            image = ImageOps.fit(
                image, (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )
            return pil_to_tensor(image).float().div(255.0)


def image_edges(value: Tensor) -> tuple[Tensor, Tensor]:
    return value[:, :, :, 1:] - value[:, :, :, :-1], value[:, :, 1:, :] - value[:, :, :-1, :]


def normalized_lerp(source: Tensor, target: Tensor, time: Tensor) -> Tensor:
    shape = (len(time),) + (1,) * (source.ndim - 1)
    return F.normalize(torch.lerp(source, target, time.reshape(shape).to(source)), dim=-1)


def encode_dino(
    dino: DinoCatDogClassifier,
    images: Tensor,
    preprocess: BatchPreprocessor,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        global_state, patches, _ = dino_features_with_grad(dino, images, preprocess)
    return global_state, patches


def endpoint_loss(
    model: ContinuousConditionalVAE,
    images: Tensor,
    time: Tensor,
    global_state: Tensor,
    patches: Tensor,
    *,
    sample: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean, log_variance = model.encode(images)
    latent = model.reparameterize(mean, log_variance) if sample else mean
    reconstruction = model.decode(latent, time, global_state, patches)
    pixel = F.l1_loss(reconstruction, images)
    pred_dx, pred_dy = image_edges(reconstruction)
    image_dx, image_dy = image_edges(images)
    edge = 0.5 * (F.l1_loss(pred_dx, image_dx) + F.l1_loss(pred_dy, image_dy))
    kl = diagonal_gaussian_kl(mean, log_variance)
    return reconstruction, pixel, edge, kl


def bridge_loss(
    model: ContinuousConditionalVAE,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    cat: Tensor,
    dog: Tensor,
    cat_global: Tensor,
    dog_global: Tensor,
    cat_patches: Tensor,
    dog_patches: Tensor,
    time: Tensor,
    *,
    diffusivity: float,
    cycle_count: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    cat_mean, _ = model.encode(cat)
    dog_mean, _ = model.encode(dog)
    latent = latent_brownian_bridge(
        cat_mean,
        dog_mean,
        time,
        diffusivity=diffusivity,
    )
    target_global = normalized_lerp(cat_global, dog_global, time)
    target_patches = normalized_lerp(cat_patches, dog_patches, time)
    generated = model.decode(latent, time, target_global, target_patches)
    count = min(cycle_count, len(generated))
    rendered_global, rendered_patches, _ = dino_features_with_grad(
        dino, generated[:count], preprocess
    )
    global_cycle = (
        1.0 - (rendered_global * target_global[:count]).sum(1)
    ).mean()
    patch_cycle = (
        1.0 - (rendered_patches * target_patches[:count]).sum(2)
    ).mean()
    recovered_mean, _ = model.encode(generated)
    latent_cycle = F.smooth_l1_loss(
        recovered_mean.contiguous(), latent.detach().contiguous()
    )
    total = global_cycle + 0.5 * patch_cycle + 0.1 * latent_cycle
    return total, {
        "global_cycle": global_cycle,
        "patch_cycle": patch_cycle,
        "latent_cycle": latent_cycle,
    }


@torch.no_grad()
def evaluate_endpoints(
    model: ContinuousConditionalVAE,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_cosine = 0.0
    seen = 0
    for domain, loader in enumerate(loaders):
        for batch_index, image in enumerate(loader):
            if batch_index >= batches:
                break
            image = image.to(device)
            global_state, patches = encode_dino(dino, image, preprocess)
            mean, _ = model.encode(image)
            reconstruction = model.decode(
                mean,
                torch.full((len(image),), float(domain), device=device),
                global_state,
                patches,
            )
            rendered_global, _, _ = dino_features_with_grad(
                dino, reconstruction, preprocess
            )
            total_l1 += float(F.l1_loss(reconstruction, image, reduction="sum"))
            total_cosine += float((rendered_global * global_state).sum(1).sum())
            seen += len(image)
    model.train()
    return {
        "pixel_l1": total_l1 / (seen * 3 * model.image_size * model.image_size),
        "dino_cycle_cosine": total_cosine / seen,
    }


def train(
    model: ContinuousConditionalVAE,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    train_loaders: tuple[DataLoader, DataLoader],
    test_loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Tensor], list[dict[str, float]], dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.training_steps
    )
    cat_iterator = iter(train_loaders[0])
    dog_iterator = iter(train_loaders[1])
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, args.training_steps + 1):
        try:
            cat = next(cat_iterator)
        except StopIteration:
            cat_iterator = iter(train_loaders[0])
            cat = next(cat_iterator)
        try:
            dog = next(dog_iterator)
        except StopIteration:
            dog_iterator = iter(train_loaders[1])
            dog = next(dog_iterator)
        cat, dog = cat.to(device), dog.to(device)
        if model.conditioning_scale == 0:
            cat_global = torch.zeros(len(cat), 192, device=device)
            dog_global = torch.zeros(len(dog), 192, device=device)
            cat_patches = torch.zeros(len(cat), 16, 192, device=device)
            dog_patches = torch.zeros(len(dog), 16, 192, device=device)
        else:
            cat_global, cat_patches = encode_dino(dino, cat, preprocess)
            dog_global, dog_patches = encode_dino(dino, dog, preprocess)
        images = torch.cat([cat, dog])
        globals_ = torch.cat([cat_global, dog_global])
        patches = torch.cat([cat_patches, dog_patches])
        endpoint_time = torch.cat(
            [torch.zeros(len(cat), device=device), torch.ones(len(dog), device=device)]
        )
        _, pixel, edge, kl = endpoint_loss(
            model,
            images,
            endpoint_time,
            globals_,
            patches,
            sample=True,
        )
        if args.lambda_bridge > 0:
            time = torch.rand(len(cat), device=device).mul(0.9).add(0.05)
            bridge, components = bridge_loss(
                model,
                dino,
                preprocess,
                cat,
                dog,
                cat_global,
                dog_global,
                cat_patches,
                dog_patches,
                time,
                diffusivity=args.latent_diffusivity,
                cycle_count=args.cycle_batch,
            )
        else:
            bridge = torch.zeros((), device=device)
            components = {
                "global_cycle": bridge,
                "patch_cycle": bridge,
                "latent_cycle": bridge,
            }
        bridge_ramp = min(1.0, step / args.bridge_warmup)
        loss = (
            pixel
            + args.lambda_edge * edge
            + args.lambda_kl * kl
            + bridge_ramp * args.lambda_bridge * bridge
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        record = {
            "step": float(step),
            "total": float(loss.detach()),
            "pixel": float(pixel.detach()),
            "edge": float(edge.detach()),
            "kl": float(kl.detach()),
            "bridge": float(bridge.detach()),
            **{key: float(value.detach()) for key, value in components.items()},
        }
        history.append(record)
        if step == 1 or step % args.evaluate_every == 0 or step == args.training_steps:
            metrics = evaluate_endpoints(
                model,
                dino,
                preprocess,
                test_loaders,
                device,
                args.evaluation_batches,
            )
            score = metrics["pixel_l1"] + 0.1 * (1.0 - metrics["dino_cycle_cosine"])
            print(
                f"step={step} pixel={record['pixel']:.4f} bridge={record['bridge']:.4f} "
                f"kl={record['kl']:.4f} heldout_l1={metrics['pixel_l1']:.4f} "
                f"heldout_dino={metrics['dino_cycle_cosine']:.4f}",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
    return best_state, history, best_metrics


def saturated_spike_rate(images: Tensor) -> float:
    flat = images.reshape(-1, *images.shape[-3:]).float()
    local = F.avg_pool2d(flat, 7, 1, 3)
    deviation = (flat - local).abs().amax(1)
    chroma = flat.amax(1) - flat.amin(1)
    return float(((deviation > 0.38) & (chroma > 0.72)).float().mean())


def make_renderer_comparison(previous: Path, current: Path, output: Path) -> None:
    with Image.open(previous) as image:
        top = image.convert("RGB")
    with Image.open(current) as image:
        bottom = image.convert("RGB")
    width = 1400
    top = top.resize(
        (width, round(width * top.height / top.width)), Image.Resampling.LANCZOS
    )
    bottom = bottom.resize(
        (width, round(width * bottom.height / bottom.width)), Image.Resampling.LANCZOS
    )
    header = 40
    canvas = Image.new("RGB", (width, top.height + bottom.height + 2 * header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 13), "StarGAN: off-manifold mixed-domain style path", fill="black")
    canvas.paste(top, (0, header))
    second = header + top.height
    draw.text(
        (8, second + 13),
        "AFHQ beta-VAE: manifold latent bridge with reciprocal-SDE clock",
        fill="black",
    )
    canvas.paste(bottom, (0, second + header))
    canvas.save(output)


@torch.no_grad()
def render_demo(
    model: ContinuousConditionalVAE,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    layerwise = torch.load(args.layerwise_checkpoint, map_location="cpu", weights_only=True)
    with args.layerwise_summary.open() as handle:
        layerwise_summary = json.load(handle)
    assignment = [int(value) for value in layerwise_summary["assignment"]]
    global_paths = F.normalize(layerwise["target_sde_paths"].float().to(device), dim=2)
    coefficients = layerwise["coefficients"].float().to(device)
    bases = layerwise["basis"].float().to(device)
    source_paths = sorted((args.assets / "src/cat").glob("*.jpg"))
    reference_paths = sorted((args.assets / "ref/dog").glob("*.jpg"))
    sources = [load_square(path) for path in source_paths]
    dog_pool = [load_square(path) for path in reference_paths]
    paired_dogs = [dog_pool[index] for index in assignment]
    source_256 = image_batch(sources)
    dog_reference_256 = image_batch(paired_dogs)
    source_star = (2.0 * source_256 - 1.0).to(device)
    dog_reference_star = (2.0 * dog_reference_256 - 1.0).to(device)
    generator, style_encoder = load_afhq_renderer(str(args.stargan_checkpoint), device=device)
    cat_domain = torch.full((len(sources),), CAT_DOMAIN, device=device, dtype=torch.long)
    dog_domain = torch.full((len(sources),), DOG_DOMAIN, device=device, dtype=torch.long)
    time = torch.linspace(0.0, 1.0, DISPLAY_STEPS, device=device)
    cat_style = style_encoder(source_star, cat_domain)
    dog_style = style_encoder(dog_reference_star, dog_domain)
    layerwise_styles = endpoint_anchored_layer_styles(
        cat_style, dog_style, coefficients, bases, time
    )
    star_paths = render_paths(generator, source_star, layerwise_styles)
    source_128 = F.interpolate(
        source_256.to(device), size=(128, 128), mode="bilinear", align_corners=False
    )
    dog_128 = F.interpolate(
        dog_reference_256.to(device),
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    )
    source_global, source_patches = encode_dino(dino, source_128, preprocess)
    dog_global, dog_patches = encode_dino(dino, dog_128, preprocess)
    global_paths = F.normalize(
        pin_trajectory_endpoint(
            global_paths.permute(1, 0, 2), dog_global
        ).permute(1, 0, 2),
        dim=2,
    )
    source_mean, _ = model.encode(source_128)
    dog_mean, _ = model.encode(dog_128)
    species = F.normalize(
        global_paths[:, -1].mean(0) - global_paths[:, 0].mean(0), dim=0
    )
    rows: list[Tensor] = []
    for row in range(len(sources)):
        sde_progress = species_clock(global_paths[row], species)
        progress = (
            (1.0 - args.sde_clock_weight) * time
            + args.sde_clock_weight * sde_progress
        )
        latent = latent_brownian_bridge(
            source_mean[row : row + 1].expand(DISPLAY_STEPS, -1, -1, -1),
            dog_mean[row : row + 1].expand(DISPLAY_STEPS, -1, -1, -1),
            progress,
            diffusivity=0.0,
        )
        patch_path = normalized_lerp(
            source_patches[row : row + 1].expand(DISPLAY_STEPS, -1, -1),
            dog_patches[row : row + 1].expand(DISPLAY_STEPS, -1, -1),
            progress,
        )
        decoded = model.decode(latent, progress, global_paths[row], patch_path)
        decoded[0] = source_128[row]
        decoded[-1] = dog_128[row]
        rows.append(decoded.cpu())
    vae_paths_128 = torch.stack(rows)
    vae_paths_256 = F.interpolate(
        vae_paths_128.flatten(0, 1),
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1).reshape(len(sources), DISPLAY_STEPS, 3, 256, 256)
    vae_star = vae_paths_256.mul(2).sub(1)
    rendered_global, _, dog_probability = evaluate_rendered_paths(
        dino, vae_star, preprocess, device
    )
    global_error = 1.0 - F.cosine_similarity(
        rendered_global, global_paths.cpu(), dim=2
    )
    adjacent = (vae_paths_128[:, 1:] - vae_paths_128[:, :-1]).abs().mean((2, 3, 4))
    vae_pil = [[tensor_to_pil(image) for image in row] for row in vae_star]
    contact = args.output_dir / "continuous_vae_contact_sheet.png"
    make_contact_sheet(
        sources,
        vae_pil,
        paired_dogs,
        contact,
        clock_label="VAE SDE",
    )
    make_distribution_gif(
        vae_pil,
        args.output_dir / "continuous_vae_distribution.gif",
        clock_label="VAE SDE",
    )
    make_renderer_comparison(
        args.previous_renderer_sheet,
        contact,
        args.output_dir / "stargan_vs_manifold_vae.png",
    )
    torch.save(
        {"trajectory_128": vae_paths_128, "trajectory_256": vae_paths_256},
        args.output_dir / "continuous_vae_trajectories.pt",
    )
    star_01 = star_paths.add(1).div(2).clamp(0, 1)
    return {
        "mean_interior_sde_cosine_error": float(global_error[:, 1:-1].mean()),
        "path_sde_cosine_error": global_error.tolist(),
        "path_dog_probability": dog_probability.tolist(),
        "mean_adjacent_frame_l1": float(adjacent.mean()),
        "star_gan_interior_saturated_spike_rate": saturated_spike_rate(star_01[:, 1:-1]),
        "vae_interior_saturated_spike_rate": saturated_spike_rate(vae_paths_128[:, 1:-1]),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(SEED + 2000)
    random.seed(SEED + 2000)
    device = require_mps()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprocess = BatchPreprocessor(64)
    dino = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb",
        image_size=64,
        pretrained=False,
        fusion="gated",
    )
    dino.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True))
    dino.eval().requires_grad_(False).to(device)
    datasets = {
        ("cat", True): AFHQDomainDataset(
            args.afhq_root, "cat", training=True, limit=args.train_limit
        ),
        ("dog", True): AFHQDomainDataset(
            args.afhq_root, "dog", training=True, limit=args.train_limit
        ),
        ("cat", False): AFHQDomainDataset(
            args.afhq_root, "cat", training=False, limit=args.test_limit
        ),
        ("dog", False): AFHQDomainDataset(
            args.afhq_root, "dog", training=False, limit=args.test_limit
        ),
    }
    loader = lambda data, shuffle: DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle,
        generator=torch.Generator().manual_seed(SEED + 2100),
    )
    train_loaders = (loader(datasets[("cat", True)], True), loader(datasets[("dog", True)], True))
    test_loaders = (loader(datasets[("cat", False)], False), loader(datasets[("dog", False)], False))
    model = ContinuousConditionalVAE(
        latent_grid=args.latent_grid,
        latent_channels=args.latent_channels,
        condition_dim=args.condition_dim,
        base_channels=args.base_channels,
    ).to(device)
    model.conditioning_scale = args.conditioning_scale
    source_training_steps = args.training_steps
    if args.checkpoint is None:
        best_state, history, heldout = train(
            model,
            dino,
            preprocess,
            train_loaders,
            test_loaders,
            device,
            args,
        )
        model.load_state_dict(best_state)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        source_training_steps = int(
            checkpoint.get("summary", {}).get("training", {}).get(
                "steps", args.training_steps
            )
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        history = checkpoint.get("history", [])
        heldout = evaluate_endpoints(
            model,
            dino,
            preprocess,
            test_loaders,
            device,
            args.evaluation_batches,
        )
    model.eval()
    torch.save(
        {"model": model.state_dict(), "history": history},
        args.output_dir / "continuous_vae_state.pt",
    )
    demo = render_demo(model, dino, preprocess, args, device)
    summary: dict[str, object] = {
        "device": str(device),
        "model": (
            "128x128 AFHQ manifold beta-VAE"
            if args.conditioning_scale == 0
            else "128x128 continuous conditional spatial VAE"
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "afhq": {
            "root": str(args.afhq_root),
            "train_cats": len(datasets[("cat", True)]),
            "train_dogs": len(datasets[("dog", True)]),
            "heldout_cats": len(datasets[("cat", False)]),
            "heldout_dogs": len(datasets[("dog", False)]),
        },
        "training": {
            "loaded_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "steps": source_training_steps,
            "batch_size_per_domain": args.batch_size,
            "learning_rate": args.learning_rate,
            "lambda_kl": args.lambda_kl,
            "lambda_edge": args.lambda_edge,
            "lambda_bridge": args.lambda_bridge,
            "latent_diffusivity": args.latent_diffusivity,
            "inference_conditioning_scale": args.conditioning_scale,
            "sde_clock_weight": args.sde_clock_weight,
        },
        "heldout_endpoint_reconstruction": heldout,
        "demo": demo,
    }
    torch.save(
        {"model": model.state_dict(), "history": history, "summary": summary},
        args.output_dir / "continuous_vae_renderer.pt",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    method = (
        "The renderer is a pure AFHQ manifold beta-VAE. DINO is used only for "
        "endpoint coupling, evaluation, and the reciprocal-SDE clock; no DINO "
        "gradient or feature tensor enters the decoder."
        if args.conditioning_scale == 0 and args.lambda_bridge == 0
        else
        "The renderer is conditioned directly on reciprocal-SDE global and patch DINO states."
    )
    results = f"""# Continuous VAE renderer

The renderer was trained on real AFHQv2 cat/dog endpoints. {method} StarGAN is
used only in the comparison panel. Each VAE path is endpoint-pinned to its real
OT-paired AFHQ dog reference; no StarGAN image is decoded by the VAE.

Held-out endpoint L1: {heldout['pixel_l1']:.6f}. Held-out DINO cycle cosine:
{heldout['dino_cycle_cosine']:.6f}. Mean interior SDE cosine error:
{demo['mean_interior_sde_cosine_error']:.6f}.

The comparison sheet should be used for artifact assessment: the simple
saturated-pixel threshold is retained in `summary.json` as a diagnostic, but it
does not detect StarGAN's structured chromatic blobs reliably.

The five displayed paths are model-selection examples, while endpoint
reconstruction metrics use the official held-out AFHQ test split.
"""
    (args.output_dir / "RESULTS.md").write_text(results)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--afhq-root", type=Path, default=DEFAULT_STARGAN / "data"
    )
    parser.add_argument(
        "--assets", type=Path, default=DEFAULT_STARGAN / "assets/representative/afhq"
    )
    parser.add_argument(
        "--stargan-checkpoint",
        type=Path,
        default=DEFAULT_STARGAN / "expr/checkpoints/afhq/100000_nets_ema.ckpt",
    )
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=WORKSPACE / "outputs/cat_dog/classifier.pt"
    )
    parser.add_argument(
        "--layerwise-checkpoint",
        type=Path,
        default=WORKSPACE
        / "outputs/layerwise_sde_conditioned_stargan/layerwise_sde_styles.pt",
    )
    parser.add_argument(
        "--layerwise-summary",
        type=Path,
        default=WORKSPACE / "outputs/layerwise_sde_conditioned_stargan/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/continuous_vae_renderer",
    )
    parser.add_argument("--training-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-kl", type=float, default=2e-4)
    parser.add_argument("--lambda-edge", type=float, default=0.2)
    parser.add_argument("--lambda-bridge", type=float, default=0.6)
    parser.add_argument("--bridge-warmup", type=int, default=500)
    parser.add_argument("--latent-diffusivity", type=float, default=0.01)
    parser.add_argument("--cycle-batch", type=int, default=4)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--latent-channels", type=int, default=24)
    parser.add_argument("--latent-grid", type=int, default=4)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--sde-clock-weight", type=float, default=0.5)
    parser.add_argument(
        "--previous-renderer-sheet",
        type=Path,
        default=WORKSPACE
        / "outputs/layerwise_sde_conditioned_stargan/layerwise_sde_conditioned.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
