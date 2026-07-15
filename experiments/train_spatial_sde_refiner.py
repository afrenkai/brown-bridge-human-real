"""Continue the SDE image refiner with spatially lifted DINO supervision."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

from brown_bridge.conditioning import cosine_bridge_loss, endpoint_anchored_layer_styles
from brown_bridge.data import BatchPreprocessor
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.refiner import SDEConditionedRefiner, deformation_temporal_loss
from brown_bridge.spatial_bridge import (
    foreground_patch_mask,
    lift_patch_trajectory,
    masked_spatial_bridge_loss,
    positional_patch_transport,
    species_clock,
)
from brown_bridge.stargan import load_afhq_renderer
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
from train_sde_refiner import refine_row, render_refined_paths


def aggregate_fit(
    model: SDEConditionedRefiner,
    dino: DinoCatDogClassifier,
    preprocess: BatchPreprocessor,
    base_paths: Tensor,
    source_images: Tensor,
    global_targets: Tensor,
    patch_targets: Tensor,
    source_patches: Tensor,
    masks: Tensor,
    time: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Return global and masked-spatial errors on the five fitted paths."""
    global_values: list[Tensor] = []
    spatial_values: list[Tensor] = []
    with torch.no_grad():
        for row in range(len(base_paths)):
            images, _, _ = refine_row(
                model,
                base_paths[row, 1:-1].to(device),
                source_images[row],
                global_targets[row, 1:-1],
                time[1:-1],
                source_state=global_targets[row, 0],
                target_state=global_targets[row, -1],
            )
            features, patches, _ = dino_features_with_grad(
                dino, images.add(1).div(2).clamp(0, 1), preprocess
            )
            global_values.append(
                1.0
                - F.cosine_similarity(features, global_targets[row, 1:-1], dim=1)
            )
            spatial_values.append(
                masked_spatial_bridge_loss(
                    patches,
                    patch_targets[row, 1:-1].to(device),
                    source_patches[row].to(device),
                    masks[row].to(device),
                )
            )
    return float(torch.cat(global_values).mean()), float(torch.stack(spatial_values).mean())


def spatial_error_components(
    rendered: Tensor,
    targets: Tensor,
    sources: Tensor,
    masks: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-path foreground-to-target and background-to-source cosine errors."""
    rendered = F.normalize(rendered.float(), dim=3)
    targets = F.normalize(targets.float(), dim=3)
    sources = F.normalize(sources.float(), dim=2)[:, None]
    target_error = 1.0 - (rendered * targets).sum(3)
    source_error = 1.0 - (rendered * sources).sum(3)
    mask = masks[:, None]
    foreground = (mask * target_error).sum((1, 2)) / (
        mask.sum((1, 2)).clamp_min(1e-6) * rendered.shape[1]
    )
    background_weight = 1.0 - mask
    background = (background_weight * source_error).sum((1, 2)) / (
        background_weight.sum((1, 2)).clamp_min(1e-6) * rendered.shape[1]
    )
    return foreground, background


def make_mask_visualization(
    sources: list[Image.Image], masks: Tensor, output: Path
) -> None:
    tile = 256
    header = 42
    canvas = Image.new("RGB", (len(sources) * tile, tile + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 13),
        "Soft DINO foreground masks (red = transported toward dog; clear = source-preserved)",
        fill="black",
    )
    for column, (source, mask) in enumerate(zip(sources, masks)):
        side = int(mask.numel() ** 0.5)
        mask_image = Image.fromarray(
            mask.reshape(side, side).clamp(0, 1).mul(190).byte().numpy(),
            mode="L",
        ).resize((tile, tile), Image.Resampling.BILINEAR)
        red = Image.new("RGB", (tile, tile), (255, 35, 35))
        overlay = Image.composite(red, source, mask_image)
        canvas.paste(overlay, (column * tile, header))
    canvas.save(output)


def append_comparison(
    previous: Path, spatial_sheet: Path, output: Path, *, width: int = 1400
) -> None:
    with Image.open(previous) as image:
        top = image.convert("RGB")
    with Image.open(spatial_sheet) as image:
        bottom = image.convert("RGB")
    if top.width != width:
        top = top.resize(
            (width, round(width * top.height / top.width)), Image.Resampling.LANCZOS
        )
    bottom = bottom.resize(
        (width, round(width * bottom.height / bottom.width)), Image.Resampling.LANCZOS
    )
    header = 42
    canvas = Image.new("RGB", (width, top.height + header + bottom.height), "white")
    canvas.paste(top, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, top.height + 13),
        "5. Warm-start decoder + spatial patch OT (foreground target, background source)",
        fill="black",
    )
    canvas.paste(bottom, (0, top.height + header))
    canvas.save(output)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(SEED)
    device = require_mps()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layerwise = torch.load(args.layerwise_checkpoint, map_location="cpu", weights_only=True)
    previous_refiner = torch.load(
        args.refiner_checkpoint, map_location="cpu", weights_only=True
    )
    with args.layerwise_summary.open() as handle:
        layerwise_summary = json.load(handle)
    with args.refiner_summary.open() as handle:
        refiner_summary = json.load(handle)
    assignment = [int(value) for value in layerwise_summary["assignment"]]
    global_targets = layerwise["target_sde_paths"].float().to(device)
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
    dino.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True))
    dino.eval().requires_grad_(False).to(device)

    generator, style_encoder = load_afhq_renderer(str(args.stargan_checkpoint), device=device)
    cat_domain = torch.full((len(sources),), CAT_DOMAIN, device=device, dtype=torch.long)
    dog_domain = torch.full((len(sources),), DOG_DOMAIN, device=device, dtype=torch.long)
    with torch.no_grad():
        cat_style = style_encoder(source_star, cat_domain)
        dog_style = style_encoder(dog_star, dog_domain)
        layerwise_styles = endpoint_anchored_layer_styles(
            cat_style, dog_style, coefficients, bases, time
        )
        base_paths = render_paths(generator, source_star, layerwise_styles)
    base_global, base_patches, _ = evaluate_rendered_paths(
        dino, base_paths, preprocess, device
    )
    source_patches = base_patches[:, 0].clone()
    endpoint_patches = base_patches[:, -1].clone()
    species = F.normalize(
        global_targets[:, -1].mean(0) - global_targets[:, 0].mean(0), dim=0
    ).cpu()

    mapped_targets: list[Tensor] = []
    plans: list[Tensor] = []
    masks: list[Tensor] = []
    patch_targets: list[Tensor] = []
    progress_values: list[Tensor] = []
    for row in range(len(sources)):
        mapped, plan, _ = positional_patch_transport(
            source_patches[row],
            endpoint_patches[row],
            position_weight=args.position_weight,
            epsilon=args.ot_epsilon,
        )
        mask = foreground_patch_mask(source_patches[row], mapped, species)
        progress = species_clock(global_targets[row].cpu(), species)
        mapped_targets.append(mapped)
        plans.append(plan)
        masks.append(mask)
        progress_values.append(progress)
        patch_targets.append(lift_patch_trajectory(source_patches[row], mapped, progress))
    mapped_tensor = torch.stack(mapped_targets)
    plan_tensor = torch.stack(plans)
    mask_tensor = torch.stack(masks)
    patch_target_tensor = torch.stack(patch_targets)

    model = SDEConditionedRefiner(
        feature_dim=global_targets.shape[-1],
        condition_dim=args.condition_dim,
        max_flow=args.max_flow,
        max_residual=args.max_residual,
        output_grid_size=args.output_grid_size,
    ).to(device)
    model.load_state_dict(previous_refiner["model"], strict=True)
    warm_state = copy.deepcopy(model.state_dict())
    warm_global_fit, warm_spatial_fit = aggregate_fit(
        model,
        dino,
        preprocess,
        base_paths,
        source_star,
        global_targets,
        patch_target_tensor,
        source_patches,
        mask_tensor,
        time,
        device,
    )
    best_score = warm_global_fit + args.lambda_spatial * warm_spatial_fit
    best_state = copy.deepcopy(warm_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    history: list[dict[str, float]] = []
    order_generator = torch.Generator().manual_seed(SEED + 1200)
    order = torch.randperm(len(sources), generator=order_generator).tolist()
    for step in range(1, args.training_steps + 1):
        if (step - 1) % len(sources) == 0:
            order = torch.randperm(len(sources), generator=order_generator).tolist()
        row = order[(step - 1) % len(sources)]
        images, flow, residual = refine_row(
            model,
            base_paths[row, 1:-1].to(device),
            source_star[row],
            global_targets[row, 1:-1],
            time[1:-1],
            source_state=global_targets[row, 0],
            target_state=global_targets[row, -1],
        )
        features, patches, _ = dino_features_with_grad(
            dino, images.add(1).div(2).clamp(0, 1), preprocess
        )
        global_loss = cosine_bridge_loss(features, global_targets[row, 1:-1])
        spatial_loss = masked_spatial_bridge_loss(
            patches,
            patch_target_tensor[row, 1:-1].to(device),
            source_patches[row].to(device),
            mask_tensor[row].to(device),
        )
        zero_flow = torch.zeros_like(flow[:1])
        zero_residual = torch.zeros_like(residual[:1])
        full_flow = torch.cat([zero_flow, flow, zero_flow], dim=0)[None]
        full_residual = torch.cat(
            [zero_residual, residual, zero_residual], dim=0
        )[None]
        temporal_loss = deformation_temporal_loss(full_flow, full_residual)
        loss = (
            global_loss
            + args.lambda_spatial * spatial_loss
            + args.lambda_temporal * temporal_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": float(step),
            "row": float(row),
            "total": float(loss.detach()),
            "global": float(global_loss.detach()),
            "spatial": float(spatial_loss.detach()),
            "temporal": float(temporal_loss.detach()),
        }
        history.append(record)
        if step == 1 or step % args.fit_every == 0 or step == args.training_steps:
            global_fit, spatial_fit = aggregate_fit(
                model,
                dino,
                preprocess,
                base_paths,
                source_star,
                global_targets,
                patch_target_tensor,
                source_patches,
                mask_tensor,
                time,
                device,
            )
            score = global_fit + args.lambda_spatial * spatial_fit
            print(
                f"step={step} aggregate_global={global_fit:.4f} "
                f"aggregate_spatial={spatial_fit:.4f} score={score:.4f} "
                f"temporal={record['temporal']:.6f}",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {"model": model.state_dict(), "history": history},
        args.output_dir / "spatial_sde_refiner_state.pt",
    )

    warm_model = copy.deepcopy(model)
    warm_model.load_state_dict(warm_state)
    warm_model.eval()
    warm_paths, _, _ = render_refined_paths(
        warm_model, base_paths, source_star, global_targets, time, device
    )
    spatial_paths, flows, residuals = render_refined_paths(
        model, base_paths, source_star, global_targets, time, device
    )
    warm_global, warm_patches, warm_probability = evaluate_rendered_paths(
        dino, warm_paths, preprocess, device
    )
    spatial_global, spatial_patches, spatial_probability = evaluate_rendered_paths(
        dino, spatial_paths, preprocess, device
    )
    target_cpu = global_targets.cpu()
    warm_global_error = 1.0 - F.cosine_similarity(warm_global, target_cpu, dim=2)
    spatial_global_error = 1.0 - F.cosine_similarity(spatial_global, target_cpu, dim=2)
    warm_foreground, warm_background = spatial_error_components(
        warm_patches[:, 1:-1],
        patch_target_tensor[:, 1:-1],
        source_patches,
        mask_tensor,
    )
    spatial_foreground, spatial_background = spatial_error_components(
        spatial_patches[:, 1:-1],
        patch_target_tensor[:, 1:-1],
        source_patches,
        mask_tensor,
    )

    spatial_pil = [
        [tensor_to_pil(image) for image in row] for row in spatial_paths
    ]
    spatial_sheet = args.output_dir / "spatial_refined_contact_sheet.png"
    make_contact_sheet(
        sources,
        spatial_pil,
        paired_dogs,
        spatial_sheet,
        clock_label="spatial SDE",
    )
    make_distribution_gif(
        spatial_pil,
        args.output_dir / "spatial_refined_distribution.gif",
        clock_label="spatial SDE",
    )
    make_mask_visualization(
        sources, mask_tensor, args.output_dir / "foreground_masks.png"
    )
    append_comparison(
        args.previous_comparison,
        spatial_sheet,
        args.output_dir / "five_way_comparison.png",
    )

    interior = slice(1, -1)
    warm_mean = warm_global_error[:, interior].mean()
    spatial_mean = spatial_global_error[:, interior].mean()
    summary: dict[str, object] = {
        "device": str(device),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": {
            "warm_started_from": str(args.refiner_checkpoint),
            "steps": args.training_steps,
            "learning_rate": args.learning_rate,
            "lambda_spatial": args.lambda_spatial,
            "lambda_temporal": args.lambda_temporal,
            "position_weight": args.position_weight,
            "ot_epsilon": args.ot_epsilon,
            "max_flow": args.max_flow,
            "max_residual": args.max_residual,
            "output_grid_size": args.output_grid_size,
            "best_aggregate_training_objective": best_score,
        },
        "warm_start_mean_interior_sde_cosine_error": float(warm_mean),
        "spatial_mean_interior_sde_cosine_error": float(spatial_mean),
        "relative_global_improvement_vs_warm_start": float(1.0 - spatial_mean / warm_mean),
        "relative_global_improvement_vs_linear": float(
            1.0
            - spatial_mean
            / float(layerwise_summary["baseline_mean_interior_sde_cosine_error"])
        ),
        "warm_start_foreground_target_error": warm_foreground.tolist(),
        "spatial_foreground_target_error": spatial_foreground.tolist(),
        "warm_start_background_source_error": warm_background.tolist(),
        "spatial_background_source_error": spatial_background.tolist(),
        "warm_start_dog_probability": warm_probability.tolist(),
        "spatial_dog_probability": spatial_probability.tolist(),
        "warm_start_path_sde_cosine_error": warm_global_error.tolist(),
        "spatial_path_sde_cosine_error": spatial_global_error.tolist(),
        "species_progress": [value.tolist() for value in progress_values],
        "foreground_masks": mask_tensor.tolist(),
        "patch_transport_plans": plan_tensor.tolist(),
        "mean_absolute_flow": float(flows.abs().mean()),
        "max_absolute_flow": float(flows.abs().max()),
        "mean_absolute_rgb_residual": float(residuals.abs().mean()),
        "max_absolute_rgb_residual": float(residuals.abs().max()),
        "previous_refiner_global_error": refiner_summary[
            "refined_mean_interior_sde_cosine_error"
        ],
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
            "summary": summary,
            "target_sde_paths": target_cpu,
            "target_patch_paths": patch_target_tensor,
            "mapped_target_patches": mapped_tensor,
            "foreground_masks": mask_tensor,
            "patch_transport_plans": plan_tensor,
        },
        args.output_dir / "spatial_sde_refiner.pt",
    )
    results = f"""# Spatial reciprocal-SDE image refinement

The global bridge target is supplemented by a positional entropic-OT lift of the
reachable dog endpoint onto the source cat's 4×4 DINO patch grid. A soft mask
transports central, species-changing patches and preserves source-like background.

The fitted objective is

`L = L_global + {args.lambda_spatial:g} L_spatial + {args.lambda_temporal:g} L_temporal`.

Mean interior global SDE cosine error changed from {float(warm_mean):.6f} to
{float(spatial_mean):.6f}. Mean foreground target error changed from
{float(warm_foreground.mean()):.6f} to {float(spatial_foreground.mean()):.6f};
background source error changed from {float(warm_background.mean()):.6f} to
{float(spatial_background.mean()):.6f}.

The colored pixel spikes are already present in the unoptimized linear StarGAN
style path. They persist with an identically zero RGB residual, identifying an
off-manifold cross-domain style interpolation failure in the fixed renderer.
The band-limited flow prevents a new pixel attack channel but cannot remove
artifacts already present in the base render.

Spatial training primarily improves source-background preservation; the
foreground-target metric must be checked separately rather than inferred from
the combined masked objective.

These are optimization-set diagnostics on five paths, not held-out estimates.
"""
    (args.output_dir / "RESULTS.md").write_text(results)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--refiner-checkpoint",
        type=Path,
        default=WORKSPACE / "outputs/sde_intermediate_decoder/sde_refiner.pt",
    )
    parser.add_argument(
        "--refiner-summary",
        type=Path,
        default=WORKSPACE / "outputs/sde_intermediate_decoder/summary.json",
    )
    parser.add_argument(
        "--previous-comparison",
        type=Path,
        default=WORKSPACE / "outputs/sde_intermediate_decoder/four_way_comparison.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/spatial_sde_intermediate_decoder",
    )
    parser.add_argument("--training-steps", type=int, default=300)
    parser.add_argument("--fit-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-spatial", type=float, default=0.7)
    parser.add_argument("--lambda-temporal", type=float, default=12.0)
    parser.add_argument("--position-weight", type=float, default=0.25)
    parser.add_argument("--ot-epsilon", type=float, default=0.08)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--max-flow", type=float, default=0.08)
    parser.add_argument("--max-residual", type=float, default=0.0)
    parser.add_argument("--output-grid-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
