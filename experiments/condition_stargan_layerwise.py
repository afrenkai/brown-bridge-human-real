"""Low-rank per-decoder-block conditioning on reciprocal DINO SDE states."""

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
    endpoint_anchored_layer_styles,
    endpoint_anchored_styles,
    orthogonal_patch_loss,
    temporal_curvature_loss,
)
from brown_bridge.data import BatchPreprocessor
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.stargan import Generator, load_afhq_renderer
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


def svd_warm_start(
    global_correction: Tensor, *, rank: int, layers: int
) -> tuple[Tensor, Tensor, float]:
    """Factor a global correction and repeat its coefficients across layers."""
    left, singular, right = torch.linalg.svd(
        global_correction.float(), full_matrices=False
    )
    actual_rank = min(rank, len(singular))
    basis = right[:actual_rank]
    time_coefficients = left[:, :actual_rank] * singular[:actual_rank]
    coefficients = time_coefficients[:, None].expand(-1, layers, -1).clone()
    reconstruction = time_coefficients @ basis
    relative_error = float(
        torch.linalg.vector_norm(reconstruction - global_correction)
        / torch.linalg.vector_norm(global_correction).clamp_min(1e-8)
    )
    return coefficients, basis, relative_error


def optimize_one_layerwise_path(
    *,
    row: int,
    generator: Generator,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    source_image: Tensor,
    source_style: Tensor,
    dog_style: Tensor,
    source_patches: Tensor,
    target_path: Tensor,
    species: Tensor,
    time: Tensor,
    initial_coefficients: Tensor,
    basis: Tensor,
    steps: int,
    learning_rate: float,
    lambda_content: float,
    lambda_temporal: float,
    max_coefficient: float,
) -> tuple[Tensor, list[dict[str, float]]]:
    coefficients = nn.Parameter(initial_coefficients[None].to(source_image))
    fixed_basis = basis[None].to(source_image)
    optimizer = torch.optim.Adam([coefficients], lr=learning_rate)
    best_loss = float("inf")
    best_coefficients = coefficients.detach().clone()
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        styles = endpoint_anchored_layer_styles(
            source_style[None],
            dog_style[None],
            coefficients,
            fixed_basis,
            time,
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
        torch.nn.utils.clip_grad_norm_([coefficients], 2.0)
        optimizer.step()
        with torch.no_grad():
            coefficients.clamp_(-max_coefficient, max_coefficient)
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
            best_coefficients = coefficients.detach().clone()
        if step == 1 or step % 20 == 0 or step == steps:
            print(
                f"path={row} step={step} total={record['total']:.4f} "
                f"bridge={record['bridge']:.4f} content={record['content']:.4f} "
                f"temporal={record['temporal']:.6f}",
                flush=True,
            )
    return best_coefficients[0].cpu(), history


def patch_content_errors(
    patches: Tensor, source_patches: Tensor, species: Tensor
) -> Tensor:
    unit = F.normalize(species.float().cpu(), dim=0)
    source = source_patches[:, None]
    projection = torch.einsum("btpd,d->btp", patches, unit)[..., None] * unit
    rendered = F.normalize(patches - projection, dim=3)
    source_projection = torch.einsum("btpd,d->btp", source, unit)[..., None] * unit
    source = F.normalize(source - source_projection, dim=3)
    return (1.0 - (rendered * source).sum(3)).mean((1, 2))


def make_three_way_comparison(
    paths: list[tuple[str, Path]], output: Path, *, width: int = 1400
) -> None:
    images: list[tuple[str, Image.Image]] = []
    for label, path in paths:
        with Image.open(path) as image:
            value = image.convert("RGB")
        value = value.resize(
            (width, round(width * value.height / value.width)),
            Image.Resampling.LANCZOS,
        )
        images.append((label, value))
    header = 38
    height = sum(image.height + header for _, image in images)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    top = 0
    for label, image in images:
        draw.text((8, top + 12), label, fill="black")
        top += header
        canvas.paste(image, (0, top))
        top += image.height
    canvas.save(output)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(SEED)
    device = require_mps()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = torch.load(
        args.global_conditioning_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    with args.global_conditioning_summary.open() as handle:
        previous_summary = json.load(handle)
    assignment = [int(value) for value in previous_summary["assignment"]]
    target_paths = previous["target_sde_paths"].float().to(device)
    global_correction = previous["style_correction"].float().to(device)

    source_paths = sorted((args.assets / "src/cat").glob("*.jpg"))
    reference_paths = sorted((args.assets / "ref/dog").glob("*.jpg"))
    sources = [load_square(path) for path in source_paths]
    dog_pool = [load_square(path) for path in reference_paths]
    paired_dogs = [dog_pool[index] for index in assignment]
    source_tensor = image_batch(sources)
    paired_dog_tensor = image_batch(paired_dogs)

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
        _, source_patches_device, _ = dino_features_with_grad(
            dino, source_tensor.to(device), preprocess
        )
    source_patches = source_patches_device.cpu()

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
    zero = torch.zeros_like(global_correction)
    baseline_styles = endpoint_anchored_styles(cat_style, dog_style, zero, time)
    global_styles = endpoint_anchored_styles(
        cat_style, dog_style, global_correction, time
    )
    baseline_paths = render_paths(generator, source_star, baseline_styles)
    global_paths = render_paths(generator, source_star, global_styles)

    layers = len(generator.decode)
    coefficient_rows: list[Tensor] = []
    basis_rows: list[Tensor] = []
    reconstruction_errors: list[float] = []
    histories: list[list[dict[str, float]]] = []
    for row in range(len(sources)):
        initial, basis, reconstruction_error = svd_warm_start(
            global_correction[row].cpu(), rank=args.rank, layers=layers
        )
        coefficients, history = optimize_one_layerwise_path(
            row=row,
            generator=generator,
            dino=dino,
            preprocess=preprocess,
            source_image=source_star[row],
            source_style=cat_style[row],
            dog_style=dog_style[row],
            source_patches=source_patches[row : row + 1].to(device),
            target_path=target_paths[row],
            species=previous_summary_species(previous_summary, previous, device),
            time=time,
            initial_coefficients=initial,
            basis=basis,
            steps=args.optimization_steps,
            learning_rate=args.learning_rate,
            lambda_content=args.lambda_content,
            lambda_temporal=args.lambda_temporal,
            max_coefficient=args.max_coefficient,
        )
        coefficient_rows.append(coefficients)
        basis_rows.append(basis)
        reconstruction_errors.append(reconstruction_error)
        histories.append(history)
        torch.mps.empty_cache()

    coefficients = torch.stack(coefficient_rows).to(device)
    bases = torch.stack(basis_rows).to(device)
    layerwise_styles = endpoint_anchored_layer_styles(
        cat_style, dog_style, coefficients, bases, time
    )
    layerwise_paths = render_paths(generator, source_star, layerwise_styles)

    baseline_global, baseline_patch, baseline_probability = evaluate_rendered_paths(
        dino, baseline_paths, preprocess, device
    )
    global_global, global_patch, global_probability = evaluate_rendered_paths(
        dino, global_paths, preprocess, device
    )
    layer_global, layer_patch, layer_probability = evaluate_rendered_paths(
        dino, layerwise_paths, preprocess, device
    )
    target_cpu = target_paths.cpu()
    baseline_error = 1.0 - F.cosine_similarity(baseline_global, target_cpu, dim=2)
    global_error = 1.0 - F.cosine_similarity(global_global, target_cpu, dim=2)
    layer_error = 1.0 - F.cosine_similarity(layer_global, target_cpu, dim=2)
    species = previous_summary_species(previous_summary, previous, device).cpu()
    baseline_content = patch_content_errors(baseline_patch, source_patches, species)
    global_content = patch_content_errors(global_patch, source_patches, species)
    layer_content = patch_content_errors(layer_patch, source_patches, species)

    def as_pil(paths: Tensor) -> list[list[Image.Image]]:
        return [[tensor_to_pil(image) for image in row] for row in paths]

    baseline_pil = as_pil(baseline_paths)
    global_pil = as_pil(global_paths)
    layer_pil = as_pil(layerwise_paths)
    baseline_sheet = args.output_dir / "linear_style_baseline.png"
    global_sheet = args.output_dir / "global_sde_conditioned.png"
    layer_sheet = args.output_dir / "layerwise_sde_conditioned.png"
    make_contact_sheet(
        sources, baseline_pil, paired_dogs, baseline_sheet, clock_label="style"
    )
    make_contact_sheet(
        sources, global_pil, paired_dogs, global_sheet, clock_label="global SDE"
    )
    make_contact_sheet(
        sources, layer_pil, paired_dogs, layer_sheet, clock_label="layer SDE"
    )
    make_distribution_gif(
        layer_pil,
        args.output_dir / "layerwise_sde_conditioned.gif",
        clock_label="layer SDE",
    )
    make_three_way_comparison(
        [
            ("1. Linear shared style", baseline_sheet),
            ("2. Global reciprocal-SDE-conditioned style", global_sheet),
            (f"3. Rank-{args.rank} per-block reciprocal-SDE conditioning", layer_sheet),
        ],
        args.output_dir / "three_way_comparison.png",
    )

    interior = slice(1, -1)
    baseline_mean = baseline_error[:, interior].mean()
    global_mean = global_error[:, interior].mean()
    layer_mean = layer_error[:, interior].mean()
    summary: dict[str, object] = {
        "device": str(device),
        "rank": args.rank,
        "decoder_blocks": layers,
        "optimization": {
            "steps_per_path": args.optimization_steps,
            "learning_rate": args.learning_rate,
            "lambda_content": args.lambda_content,
            "lambda_temporal": args.lambda_temporal,
            "max_coefficient": args.max_coefficient,
        },
        "assignment": assignment,
        "paired_references": [reference_paths[index].name for index in assignment],
        "svd_warm_start_relative_errors": reconstruction_errors,
        "baseline_mean_interior_sde_cosine_error": float(baseline_mean),
        "global_mean_interior_sde_cosine_error": float(global_mean),
        "layerwise_mean_interior_sde_cosine_error": float(layer_mean),
        "layerwise_relative_improvement_vs_baseline": float(
            1.0 - layer_mean / baseline_mean
        ),
        "layerwise_relative_improvement_vs_global": float(
            1.0 - layer_mean / global_mean
        ),
        "baseline_path_dog_probability": baseline_probability.tolist(),
        "global_path_dog_probability": global_probability.tolist(),
        "layerwise_path_dog_probability": layer_probability.tolist(),
        "baseline_content_error": baseline_content.tolist(),
        "global_content_error": global_content.tolist(),
        "layerwise_content_error": layer_content.tolist(),
        "baseline_path_sde_cosine_error": baseline_error.tolist(),
        "global_path_sde_cosine_error": global_error.tolist(),
        "layerwise_path_sde_cosine_error": layer_error.tolist(),
        "final_training_losses": [history[-1] for history in histories],
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(
        {
            "coefficients": coefficients.cpu(),
            "basis": bases.cpu(),
            "layerwise_styles": layerwise_styles.cpu(),
            "target_sde_paths": target_cpu,
            "histories": histories,
            "summary": summary,
        },
        args.output_dir / "layerwise_sde_styles.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def previous_summary_species(
    summary: dict[str, object], checkpoint: dict[str, object], device: torch.device
) -> Tensor:
    # The style-conditioning artifact intentionally does not duplicate the
    # reciprocal checkpoint. Its target paths live in the same trained DINO
    # coordinates; recover the normalized empirical endpoint direction here.
    target = checkpoint["target_sde_paths"].float()
    direction = target[:, -1].mean(0) - target[:, 0].mean(0)
    return F.normalize(direction, dim=0).to(device)


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
        "--global-conditioning-checkpoint",
        type=Path,
        default=WORKSPACE
        / "outputs/sde_conditioned_stargan/sde_conditioned_styles.pt",
    )
    parser.add_argument(
        "--global-conditioning-summary",
        type=Path,
        default=WORKSPACE / "outputs/sde_conditioned_stargan/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/layerwise_sde_conditioned_stargan",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--optimization-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--lambda-content", type=float, default=0.35)
    parser.add_argument("--lambda-temporal", type=float, default=8.0)
    parser.add_argument("--max-coefficient", type=float, default=6.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
