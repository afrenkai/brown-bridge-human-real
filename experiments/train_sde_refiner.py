"""Train a small RGB refiner explicitly on reciprocal-DINO intermediate states."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

from brown_bridge.conditioning import (
    cosine_bridge_loss,
    endpoint_anchored_layer_styles,
    orthogonal_patch_loss,
)
from brown_bridge.data import BatchPreprocessor
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.refiner import SDEConditionedRefiner, deformation_temporal_loss
from brown_bridge.stargan import load_afhq_renderer
from condition_stargan_layerwise import patch_content_errors
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


def refine_row(
    model: SDEConditionedRefiner,
    base_path: Tensor,
    source_image: Tensor,
    target_path: Tensor,
    time: Tensor,
    *,
    source_state: Tensor | None = None,
    target_state: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    count = len(time)
    source = source_image[None].expand(count, -1, -1, -1)
    start_value = target_path[0] if source_state is None else source_state
    endpoint_value = target_path[-1] if target_state is None else target_state
    start = start_value[None].expand(count, -1)
    endpoint = endpoint_value[None].expand(count, -1)
    return model(base_path, source, start, target_path, endpoint, time)


@torch.no_grad()
def render_refined_paths(
    model: SDEConditionedRefiner,
    base_paths: Tensor,
    source_images: Tensor,
    target_paths: Tensor,
    time: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    image_rows: list[Tensor] = []
    flow_rows: list[Tensor] = []
    residual_rows: list[Tensor] = []
    for row in range(len(base_paths)):
        images, flow, residual = refine_row(
            model,
            base_paths[row].to(device),
            source_images[row],
            target_paths[row],
            time,
        )
        image_rows.append(images.cpu())
        flow_rows.append(flow.cpu())
        residual_rows.append(residual.cpu())
    return torch.stack(image_rows), torch.stack(flow_rows), torch.stack(residual_rows)


@torch.no_grad()
def aggregate_bridge_error(
    model: SDEConditionedRefiner,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    base_paths: Tensor,
    source_images: Tensor,
    target_paths: Tensor,
    time: Tensor,
    device: torch.device,
) -> float:
    values: list[Tensor] = []
    for row in range(len(base_paths)):
        images, _, _ = refine_row(
            model,
            base_paths[row, 1:-1].to(device),
            source_images[row],
            target_paths[row, 1:-1],
            time[1:-1],
            source_state=target_paths[row, 0],
            target_state=target_paths[row, -1],
        )
        features, _, _ = dino_features_with_grad(
            dino, images.add(1).div(2).clamp(0, 1), preprocess
        )
        values.append(
            1.0
            - F.cosine_similarity(features, target_paths[row, 1:-1], dim=1)
        )
    return float(torch.cat(values).mean())


def make_four_way_comparison(
    previous_comparison: Path,
    refined_sheet: Path,
    output: Path,
    *,
    width: int = 1400,
) -> None:
    with Image.open(previous_comparison) as image:
        previous = image.convert("RGB")
    with Image.open(refined_sheet) as image:
        refined = image.convert("RGB")
    refined = refined.resize(
        (width, round(width * refined.height / refined.width)),
        Image.Resampling.LANCZOS,
    )
    if previous.width != width:
        previous = previous.resize(
            (width, round(width * previous.height / previous.width)),
            Image.Resampling.LANCZOS,
        )
    header = 38
    canvas = Image.new(
        "RGB", (width, previous.height + header + refined.height), "white"
    )
    canvas.paste(previous, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, previous.height + 12),
        "4. Learned warp/residual decoder conditioned on reciprocal-SDE states",
        fill="black",
    )
    canvas.paste(refined, (0, previous.height + header))
    canvas.save(output)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(SEED)
    device = require_mps()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layerwise = torch.load(
        args.layerwise_checkpoint, map_location="cpu", weights_only=True
    )
    with args.layerwise_summary.open() as handle:
        layerwise_summary = json.load(handle)
    assignment = [int(value) for value in layerwise_summary["assignment"]]
    target_paths = layerwise["target_sde_paths"].float().to(device)
    coefficients = layerwise["coefficients"].float().to(device)
    bases = layerwise["basis"].float().to(device)

    source_paths = sorted((args.assets / "src/cat").glob("*.jpg"))
    reference_paths = sorted((args.assets / "ref/dog").glob("*.jpg"))
    sources = [load_square(path) for path in source_paths]
    dog_pool = [load_square(path) for path in reference_paths]
    paired_dogs = [dog_pool[index] for index in assignment]
    source_tensor = image_batch(sources)
    paired_dog_tensor = image_batch(paired_dogs)
    source_star = (2.0 * source_tensor - 1.0).to(device)
    dog_star = (2.0 * paired_dog_tensor - 1.0).to(device)
    time = torch.linspace(0.0, 1.0, DISPLAY_STEPS, device=device)

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
    species = F.normalize(
        target_paths[:, -1].mean(0) - target_paths[:, 0].mean(0), dim=0
    )

    generator, style_encoder = load_afhq_renderer(
        str(args.stargan_checkpoint), device=device
    )
    cat_domain = torch.full(
        (len(sources),), CAT_DOMAIN, device=device, dtype=torch.long
    )
    dog_domain = torch.full(
        (len(sources),), DOG_DOMAIN, device=device, dtype=torch.long
    )
    with torch.no_grad():
        cat_style = style_encoder(source_star, cat_domain)
        dog_style = style_encoder(dog_star, dog_domain)
        layerwise_styles = endpoint_anchored_layer_styles(
            cat_style, dog_style, coefficients, bases, time
        )
        base_paths = render_paths(generator, source_star, layerwise_styles)

    model = SDEConditionedRefiner(
        feature_dim=target_paths.shape[-1],
        condition_dim=args.condition_dim,
        max_flow=args.max_flow,
        max_residual=args.max_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    history: list[dict[str, float]] = []
    best_error = aggregate_bridge_error(
        model,
        dino,
        preprocess,
        base_paths,
        source_star,
        target_paths,
        time,
        device,
    )
    best_state = copy.deepcopy(model.state_dict())
    generator_order = torch.Generator().manual_seed(SEED + 800)
    order = torch.randperm(len(sources), generator=generator_order).tolist()
    for step in range(1, args.training_steps + 1):
        if (step - 1) % len(sources) == 0:
            order = torch.randperm(
                len(sources), generator=generator_order
            ).tolist()
        row = order[(step - 1) % len(sources)]
        images, flow, residual = refine_row(
            model,
            base_paths[row, 1:-1].to(device),
            source_star[row],
            target_paths[row, 1:-1],
            time[1:-1],
            source_state=target_paths[row, 0],
            target_state=target_paths[row, -1],
        )
        features, patches, _ = dino_features_with_grad(
            dino, images.add(1).div(2).clamp(0, 1), preprocess
        )
        bridge = cosine_bridge_loss(features, target_paths[row, 1:-1])
        content = orthogonal_patch_loss(
            patches, source_patches[row : row + 1].to(device), species
        )
        zero_flow = torch.zeros_like(flow[:1])
        zero_residual = torch.zeros_like(residual[:1])
        full_flow = torch.cat([zero_flow, flow, zero_flow], dim=0)[None]
        full_residual = torch.cat(
            [zero_residual, residual, zero_residual], dim=0
        )[None]
        temporal = deformation_temporal_loss(full_flow, full_residual)
        loss = bridge + args.lambda_content * content + args.lambda_temporal * temporal
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": float(step),
            "row": float(row),
            "total": float(loss.detach()),
            "bridge": float(bridge.detach()),
            "content": float(content.detach()),
            "temporal": float(temporal.detach()),
        }
        history.append(record)
        if step == 1 or step % args.validation_every == 0 or step == args.training_steps:
            aggregate = aggregate_bridge_error(
                model,
                dino,
                preprocess,
                base_paths,
                source_star,
                target_paths,
                time,
                device,
            )
            print(
                f"step={step} aggregate_bridge={aggregate:.4f} "
                f"train_bridge={record['bridge']:.4f} content={record['content']:.4f} "
                f"temporal={record['temporal']:.6f}",
                flush=True,
            )
            if aggregate < best_error:
                best_error = aggregate
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    refined_paths, flows, residuals = render_refined_paths(
        model, base_paths, source_star, target_paths, time, device
    )
    base_global, base_patch, base_probability = evaluate_rendered_paths(
        dino, base_paths, preprocess, device
    )
    refined_global, refined_patch, refined_probability = evaluate_rendered_paths(
        dino, refined_paths, preprocess, device
    )
    target_cpu = target_paths.cpu()
    base_error = 1.0 - F.cosine_similarity(base_global, target_cpu, dim=2)
    refined_error = 1.0 - F.cosine_similarity(refined_global, target_cpu, dim=2)
    base_content = patch_content_errors(base_patch, source_patches, species.cpu())
    refined_content = patch_content_errors(
        refined_patch, source_patches, species.cpu()
    )

    refined_pil = [
        [tensor_to_pil(image) for image in row] for row in refined_paths
    ]
    refined_sheet = args.output_dir / "sde_refined_contact_sheet.png"
    make_contact_sheet(
        sources,
        refined_pil,
        paired_dogs,
        refined_sheet,
        clock_label="SDE decoder",
    )
    make_distribution_gif(
        refined_pil,
        args.output_dir / "sde_refined_distribution.gif",
        clock_label="SDE decoder",
    )
    make_four_way_comparison(
        args.previous_comparison,
        refined_sheet,
        args.output_dir / "four_way_comparison.png",
    )

    interior = slice(1, -1)
    base_mean = base_error[:, interior].mean()
    refined_mean = refined_error[:, interior].mean()
    summary: dict[str, object] = {
        "device": str(device),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "training": {
            "steps": args.training_steps,
            "learning_rate": args.learning_rate,
            "lambda_content": args.lambda_content,
            "lambda_temporal": args.lambda_temporal,
            "max_flow": args.max_flow,
            "max_residual": args.max_residual,
            "best_aggregate_training_bridge_error": best_error,
        },
        "layerwise_mean_interior_sde_cosine_error": float(base_mean),
        "refined_mean_interior_sde_cosine_error": float(refined_mean),
        "relative_improvement_vs_layerwise": float(1.0 - refined_mean / base_mean),
        "relative_improvement_vs_linear": float(
            1.0
            - refined_mean
            / float(layerwise_summary["baseline_mean_interior_sde_cosine_error"])
        ),
        "layerwise_path_dog_probability": base_probability.tolist(),
        "refined_path_dog_probability": refined_probability.tolist(),
        "layerwise_content_error": base_content.tolist(),
        "refined_content_error": refined_content.tolist(),
        "layerwise_path_sde_cosine_error": base_error.tolist(),
        "refined_path_sde_cosine_error": refined_error.tolist(),
        "mean_absolute_flow": float(flows.abs().mean()),
        "max_absolute_flow": float(flows.abs().max()),
        "mean_absolute_rgb_residual": float(residuals.abs().mean()),
        "max_absolute_rgb_residual": float(residuals.abs().max()),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
            "summary": summary,
            "target_sde_paths": target_cpu,
        },
        args.output_dir / "sde_refiner.pt",
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
        "--previous-comparison",
        type=Path,
        default=WORKSPACE
        / "outputs/layerwise_sde_conditioned_stargan/three_way_comparison.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/sde_intermediate_decoder",
    )
    parser.add_argument("--training-steps", type=int, default=300)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lambda-content", type=float, default=0.4)
    parser.add_argument("--lambda-temporal", type=float, default=12.0)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--max-flow", type=float, default=0.08)
    parser.add_argument("--max-residual", type=float, default=0.12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
