"""Invert StarGAN styles onto states sampled from the reciprocal DINO SDE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor, nn

from brown_bridge.conditioning import (
    cosine_bridge_loss,
    endpoint_anchored_styles,
    orthogonal_patch_loss,
    pin_trajectory_endpoint,
    temporal_curvature_loss,
)
from brown_bridge.data import BatchPreprocessor
from brown_bridge.features import (
    content_cost_matrix,
    content_descriptor,
    species_direction,
)
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.reciprocal import DirectionalTimeMLP, integrate_directional_sde
from brown_bridge.stargan import load_afhq_renderer
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
    unique_minimum_assignment,
)


FEATURE_DIM = 192
SDE_STEPS = 60
TRAIN_KAPPA = 0.0025


def dino_features_with_grad(
    model: DinoCatDogClassifier,
    images: Tensor,
    preprocess: BatchPreprocessor,
) -> tuple[Tensor, Tensor, Tensor]:
    # MPS does not implement the antialiased bilinear backward used by the
    # evaluation preprocessor. Use one consistent differentiable resize for
    # sources, SDE targets, and renderer outputs in this inversion experiment.
    resized = F.interpolate(
        images,
        size=(preprocess.image_size, preprocess.image_size),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    normalized = (
        resized - preprocess.mean.to(device=images.device, dtype=images.dtype)
    ) / preprocess.std.to(device=images.device, dtype=images.dtype)
    features, _, patches = model.encode_spatial(normalized)
    if patches is None:
        raise RuntimeError("DINO backbone did not return patch tokens")
    probability = model.classifier(features).softmax(1)[:, DOG_DOMAIN]
    return (
        F.normalize(features.float(), dim=1),
        F.normalize(patches.float(), dim=2),
        probability.float(),
    )


def load_reciprocal_fields(
    checkpoint_path: Path, device: torch.device
) -> tuple[DirectionalTimeMLP, DirectionalTimeMLP, Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    velocity = DirectionalTimeMLP(
        FEATURE_DIM, width=256, depth=3, time_frequencies=3
    )
    score = DirectionalTimeMLP(
        FEATURE_DIM, width=256, depth=3, time_frequencies=3
    )
    velocity.load_state_dict(checkpoint["velocity"], strict=True)
    score.load_state_dict(checkpoint["score"], strict=True)
    velocity.eval().requires_grad_(False).to(device)
    score.eval().requires_grad_(False).to(device)
    species = F.normalize(checkpoint["species_direction"].float(), dim=0).to(device)
    return velocity, score, species


@torch.no_grad()
def render_paths(
    generator: nn.Module,
    source_images: Tensor,
    styles: Tensor,
) -> Tensor:
    rows: list[Tensor] = []
    for row in range(len(source_images)):
        source = source_images[row : row + 1].expand(styles.shape[1], -1, -1, -1)
        rendered = generator(source, styles[row])
        rendered[0] = source_images[row]
        rows.append(rendered.cpu())
    return torch.stack(rows)


@torch.no_grad()
def evaluate_rendered_paths(
    model: DinoCatDogClassifier,
    paths: Tensor,
    preprocess: BatchPreprocessor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    global_rows: list[Tensor] = []
    patch_rows: list[Tensor] = []
    probability_rows: list[Tensor] = []
    for path in paths:
        images = path.to(device).add(1.0).div(2.0).clamp(0.0, 1.0)
        global_features, patches, probability = dino_features_with_grad(
            model, images, preprocess
        )
        global_rows.append(global_features.cpu())
        patch_rows.append(patches.cpu())
        probability_rows.append(probability.cpu())
    return (
        torch.stack(global_rows),
        torch.stack(patch_rows),
        torch.stack(probability_rows),
    )


def optimize_one_path(
    *,
    row: int,
    generator: nn.Module,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    source_image: Tensor,
    source_style: Tensor,
    dog_style: Tensor,
    source_patches: Tensor,
    target_path: Tensor,
    species: Tensor,
    time: Tensor,
    steps: int,
    learning_rate: float,
    lambda_content: float,
    lambda_temporal: float,
    max_correction: float,
) -> tuple[Tensor, list[dict[str, float]]]:
    correction = nn.Parameter(
        torch.zeros(1, len(time), source_style.shape[-1], device=source_image.device)
    )
    optimizer = torch.optim.Adam([correction], lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_correction = correction.detach().clone()
    for step in range(1, steps + 1):
        styles = endpoint_anchored_styles(
            source_style[None], dog_style[None], correction, time
        )
        interior_count = len(time) - 2
        images = generator(
            source_image[None].expand(interior_count, -1, -1, -1),
            styles[0, 1:-1],
        )
        images_01 = images.add(1.0).div(2.0).clamp(0.0, 1.0)
        rendered_features, rendered_patches, _ = dino_features_with_grad(
            dino, images_01, preprocess
        )
        bridge = cosine_bridge_loss(rendered_features, target_path[1:-1])
        content = orthogonal_patch_loss(
            rendered_patches, source_patches, species
        )
        temporal = temporal_curvature_loss(styles)
        loss = bridge + lambda_content * content + lambda_temporal * temporal
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([correction], 2.0)
        optimizer.step()
        with torch.no_grad():
            correction.clamp_(-max_correction, max_correction)
        record = {
            "step": float(step),
            "total": float(loss.detach()),
            "bridge": float(bridge.detach()),
            "content": float(content.detach()),
            "temporal": float(temporal.detach()),
        }
        history.append(record)
        if record["total"] < best_loss:
            best_loss = record["total"]
            best_correction = correction.detach().clone()
        if step == 1 or step % 20 == 0 or step == steps:
            print(
                f"path={row} step={step} total={record['total']:.4f} "
                f"bridge={record['bridge']:.4f} content={record['content']:.4f} "
                f"temporal={record['temporal']:.6f}",
                flush=True,
            )
    return best_correction[0].cpu(), history


def make_comparison(
    baseline_path: Path, conditioned_path: Path, output: Path
) -> None:
    with Image.open(baseline_path) as baseline_image:
        baseline = baseline_image.convert("RGB")
    with Image.open(conditioned_path) as conditioned_image:
        conditioned = conditioned_image.convert("RGB")
    width = 1400
    baseline = baseline.resize(
        (width, round(width * baseline.height / baseline.width)),
        Image.Resampling.LANCZOS,
    )
    conditioned = conditioned.resize(
        (width, round(width * conditioned.height / conditioned.width)),
        Image.Resampling.LANCZOS,
    )
    header = 38
    canvas = Image.new(
        "RGB", (width, 2 * header + baseline.height + conditioned.height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 12), "Baseline: linear StarGAN style clock", fill="black")
    canvas.paste(baseline, (0, header))
    second_top = header + baseline.height
    draw.text(
        (8, second_top + 12),
        "Conditioned: styles optimized against realized reciprocal-SDE states",
        fill="black",
    )
    canvas.paste(conditioned, (0, second_top + header))
    canvas.save(output)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(SEED)
    device = require_mps()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = sorted((args.assets / "src/cat").glob("*.jpg"))
    reference_paths = sorted((args.assets / "ref/dog").glob("*.jpg"))
    sources = [load_square(path) for path in source_paths]
    dog_pool = [load_square(path) for path in reference_paths]
    source_tensor = image_batch(sources)
    dog_tensor = image_batch(dog_pool)
    preprocess = BatchPreprocessor(64)

    dino = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb",
        image_size=64,
        pretrained=False,
        fusion="gated",
    )
    dino.load_state_dict(
        torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True)
    )
    dino.eval().requires_grad_(False).to(device)
    with torch.no_grad():
        cat_global_device, cat_patch_device, _ = dino_features_with_grad(
            dino, source_tensor.to(device), preprocess
        )
        dog_global_device, dog_patch_device, _ = dino_features_with_grad(
            dino, dog_tensor.to(device), preprocess
        )
    cat_global = cat_global_device.cpu()
    cat_patch = cat_patch_device.cpu()
    dog_global = dog_global_device.cpu()
    dog_patch = dog_patch_device.cpu()
    empirical_species = species_direction(cat_global, dog_global)
    cat_descriptor = content_descriptor(cat_global, cat_patch, empirical_species)
    dog_descriptor = content_descriptor(dog_global, dog_patch, empirical_species)
    assignment = unique_minimum_assignment(
        content_cost_matrix(cat_descriptor, dog_descriptor)
    )
    paired_dogs = [dog_pool[index] for index in assignment]
    paired_dog_tensor = dog_tensor[assignment]

    generator, style_encoder = load_afhq_renderer(
        str(args.stargan_checkpoint), device=device
    )
    source_star = (2.0 * source_tensor - 1.0).to(device)
    dog_star = (2.0 * paired_dog_tensor - 1.0).to(device)
    cat_domain = torch.full(
        (len(sources),), CAT_DOMAIN, device=device, dtype=torch.long
    )
    dog_domain = torch.full(
        (len(sources),), DOG_DOMAIN, device=device, dtype=torch.long
    )
    with torch.no_grad():
        cat_style = style_encoder(source_star, cat_domain)
        dog_style = style_encoder(dog_star, dog_domain)
    time = torch.linspace(0.0, 1.0, DISPLAY_STEPS, device=device)
    zero_correction = torch.zeros(
        len(sources), DISPLAY_STEPS, cat_style.shape[1], device=device
    )
    baseline_styles = endpoint_anchored_styles(
        cat_style, dog_style, zero_correction, time
    )
    baseline_paths = render_paths(generator, source_star, baseline_styles)
    baseline_global, baseline_patch, baseline_probability = evaluate_rendered_paths(
        dino, baseline_paths, preprocess, device
    )

    velocity, score, checkpoint_species = load_reciprocal_fields(
        args.reciprocal_checkpoint, device
    )
    initial = cat_global.to(device)
    noise_generator = torch.Generator().manual_seed(SEED + 300)
    noise = torch.randn(
        SDE_STEPS,
        len(sources),
        FEATURE_DIM,
        generator=noise_generator,
    ).to(device)
    with torch.no_grad():
        raw_sde = integrate_directional_sde(
            velocity,
            score,
            initial,
            direction=1,
            steps=SDE_STEPS,
            kappa=args.kappa,
            noise=noise,
        )
        reachable_dog_endpoint = baseline_global[:, -1].to(device)
        pinned_sde = pin_trajectory_endpoint(raw_sde, reachable_dog_endpoint)
        display_indices = torch.linspace(
            0, SDE_STEPS, DISPLAY_STEPS
        ).round().long()
        target_paths = F.normalize(
            pinned_sde[display_indices].permute(1, 0, 2), dim=2
        )

    corrections: list[Tensor] = []
    histories: list[list[dict[str, float]]] = []
    for row in range(len(sources)):
        correction, history = optimize_one_path(
            row=row,
            generator=generator,
            dino=dino,
            preprocess=preprocess,
            source_image=source_star[row],
            source_style=cat_style[row],
            dog_style=dog_style[row],
            source_patches=cat_patch[row : row + 1].to(device),
            target_path=target_paths[row],
            species=checkpoint_species,
            time=time,
            steps=args.optimization_steps,
            learning_rate=args.learning_rate,
            lambda_content=args.lambda_content,
            lambda_temporal=args.lambda_temporal,
            max_correction=args.max_correction,
        )
        corrections.append(correction)
        histories.append(history)
        torch.mps.empty_cache()

    correction_tensor = torch.stack(corrections).to(device)
    conditioned_styles = endpoint_anchored_styles(
        cat_style, dog_style, correction_tensor, time
    )
    conditioned_paths = render_paths(generator, source_star, conditioned_styles)
    conditioned_global, conditioned_patch, conditioned_probability = (
        evaluate_rendered_paths(dino, conditioned_paths, preprocess, device)
    )

    target_cpu = target_paths.cpu()
    baseline_error = 1.0 - F.cosine_similarity(
        baseline_global, target_cpu, dim=2
    )
    conditioned_error = 1.0 - F.cosine_similarity(
        conditioned_global, target_cpu, dim=2
    )
    source_content = cat_patch[:, None]
    unit = checkpoint_species.cpu()

    def patch_content_error(patches: Tensor) -> Tensor:
        projection = torch.einsum("btpd,d->btp", patches, unit)[..., None] * unit
        rendered = F.normalize(patches - projection, dim=3)
        source_projection = (
            torch.einsum("btpd,d->btp", source_content, unit)[..., None] * unit
        )
        source = F.normalize(source_content - source_projection, dim=3)
        return (1.0 - (rendered * source).sum(3)).mean((1, 2))

    baseline_content = patch_content_error(baseline_patch)
    conditioned_content = patch_content_error(conditioned_patch)

    baseline_pil = [
        [tensor_to_pil(image) for image in row] for row in baseline_paths
    ]
    conditioned_pil = [
        [tensor_to_pil(image) for image in row] for row in conditioned_paths
    ]
    baseline_sheet = args.output_dir / "linear_style_baseline.png"
    conditioned_sheet = args.output_dir / "sde_conditioned_contact_sheet.png"
    make_contact_sheet(
        sources,
        baseline_pil,
        paired_dogs,
        baseline_sheet,
        clock_label="style",
    )
    make_contact_sheet(
        sources,
        conditioned_pil,
        paired_dogs,
        conditioned_sheet,
        clock_label="SDE",
    )
    make_distribution_gif(
        conditioned_pil,
        args.output_dir / "sde_conditioned_distribution.gif",
        clock_label="SDE",
    )
    make_comparison(
        baseline_sheet,
        conditioned_sheet,
        args.output_dir / "baseline_vs_sde_conditioned.png",
    )

    interior = slice(1, -1)
    summary: dict[str, object] = {
        "device": str(device),
        "reciprocal_checkpoint": str(args.reciprocal_checkpoint),
        "sde_steps": SDE_STEPS,
        "display_steps": DISPLAY_STEPS,
        "diffusivity": args.kappa,
        "optimization": {
            "steps_per_path": args.optimization_steps,
            "learning_rate": args.learning_rate,
            "lambda_content": args.lambda_content,
            "lambda_temporal": args.lambda_temporal,
            "max_correction": args.max_correction,
        },
        "assignment": assignment,
        "paired_references": [reference_paths[index].name for index in assignment],
        "baseline_mean_interior_sde_cosine_error": float(
            baseline_error[:, interior].mean()
        ),
        "conditioned_mean_interior_sde_cosine_error": float(
            conditioned_error[:, interior].mean()
        ),
        "relative_sde_alignment_improvement": float(
            1.0
            - conditioned_error[:, interior].mean()
            / baseline_error[:, interior].mean()
        ),
        "baseline_path_sde_cosine_error": baseline_error.tolist(),
        "conditioned_path_sde_cosine_error": conditioned_error.tolist(),
        "baseline_path_dog_probability": baseline_probability.tolist(),
        "conditioned_path_dog_probability": conditioned_probability.tolist(),
        "target_sde_species_coordinate": (
            target_cpu @ checkpoint_species.cpu()
        ).tolist(),
        "baseline_content_error": baseline_content.tolist(),
        "conditioned_content_error": conditioned_content.tolist(),
        "final_training_losses": [history[-1] for history in histories],
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(
        {
            "style_correction": correction_tensor.cpu(),
            "conditioned_styles": conditioned_styles.cpu(),
            "target_sde_paths": target_cpu,
            "histories": histories,
            "summary": summary,
        },
        args.output_dir / "sde_conditioned_styles.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        type=Path,
        default=DEFAULT_STARGAN / "assets/representative/afhq",
    )
    parser.add_argument(
        "--stargan-checkpoint",
        type=Path,
        default=DEFAULT_STARGAN / "expr/checkpoints/afhq/100000_nets_ema.ckpt",
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default=WORKSPACE / "outputs/cat_dog/classifier.pt",
    )
    parser.add_argument(
        "--reciprocal-checkpoint",
        type=Path,
        default=WORKSPACE
        / "outputs/augmented_reciprocal_cat_dog/augmented_reciprocal_bridge.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/sde_conditioned_stargan",
    )
    parser.add_argument("--optimization-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--lambda-content", type=float, default=0.35)
    parser.add_argument("--lambda-temporal", type=float, default=8.0)
    parser.add_argument("--max-correction", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=TRAIN_KAPPA)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
