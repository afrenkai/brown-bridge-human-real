"""Learn a shared, physical-time layerwise observation path for style LightSB.

The LightSB endpoint conditional is frozen.  This experiment learns only how
the frozen StarGAN decoder blocks observe that endpoint chord, using AFHQ train
cats.  Evaluation uses uniformly spaced physical times and a fixed unselected
held-out gallery; no semantic re-timing or dog reference is used at inference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.layerwise_style import (
    EndpointVanishingResidualStyleController,
    MonotoneLayerwiseStyleController,
    clip_layerwise_clock,
    layerwise_clock_regularizer,
    layerwise_support_regularizer,
    localized_chroma_artifact_loss,
    residual_style_regularizers,
)
from brown_bridge.light_sb import CoordinateStandardizer, DiagonalLightSB
from brown_bridge.probes import BinaryLinearProbe
from brown_bridge.stargan import Generator, load_afhq_renderer
from condition_stargan_on_reciprocal_sde import dino_features_with_grad
from render_dino_stargan_bridge import CAT_DOMAIN, dino_encode, make_distribution_gif
from render_full_dataset_demo import fit_or_load_afhq_probe
from train_continuous_vae_renderer import AFHQDomainDataset
from train_distributional_latent_transport import endpoint_distribution_metrics
from train_stargan_style_lightsb import (
    dino_path_features,
    load_dino,
    path_metrics,
    pil_paths,
    save_probability_plot,
    save_source_only_sheet,
)


def total_variation(images: Tensor) -> Tensor:
    horizontal = (images[..., 1:] - images[..., :-1]).abs().mean((1, 2, 3))
    vertical = (images[..., 1:, :] - images[..., :-1, :]).abs().mean((1, 2, 3))
    return 0.5 * (horizontal + vertical)


def project_off_direction(features: Tensor, direction: Tensor) -> Tensor:
    unit = F.normalize(direction.to(features), dim=0)
    return F.normalize(features - (features @ unit)[..., None] * unit, dim=-1)


def load_transport(
    args: argparse.Namespace, device: torch.device
) -> tuple[DiagonalLightSB, CoordinateStandardizer]:
    payload = torch.load(args.transport_checkpoint, map_location="cpu", weights_only=False)
    centers = payload["lightsb"]["centers"]
    epsilon = float(payload["lightsb"]["epsilon"])
    model = DiagonalLightSB(
        centers.shape[1], components=centers.shape[0], epsilon=epsilon, initial_scale=0.1
    )
    model.load_state_dict(payload["lightsb"])
    return model.eval().requires_grad_(False).to(device), CoordinateStandardizer.from_state_dict(
        payload["standardizer"]
    )


def next_images(iterator: object, loader: DataLoader) -> tuple[Tensor, object]:
    try:
        images = next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        images = next(iterator)
    return images, iterator


StyleController = (
    MonotoneLayerwiseStyleController | EndpointVanishingResidualStyleController
)


def controlled_style_path(
    controller: StyleController,
    source_style: Tensor,
    target_style: Tensor,
    context: Tensor,
    times: Tensor,
    *,
    clock_tube: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply clock projection while retaining an optional residual detour."""
    if isinstance(controller, EndpointVanishingResidualStyleController):
        _, raw_clock, residual = controller(
            source_style, target_style, context, times
        )
    else:
        _, raw_clock = controller(source_style, target_style, context, times)
        residual = source_style.new_zeros(
            len(source_style), len(times), controller.layers, controller.style_dimension
        )
    clock = clip_layerwise_clock(raw_clock, clock_tube)
    chord = source_style[:, None, None] + clock[..., None] * (
        target_style - source_style
    )[:, None, None]
    return chord + residual, clock, raw_clock, residual


@torch.no_grad()
def endpoint_state(
    images_01: Tensor,
    style_encoder: torch.nn.Module,
    lightsb: DiagonalLightSB,
    standardizer: CoordinateStandardizer,
    renderer: Generator,
    dino: torch.nn.Module,
    preprocess: BatchPreprocessor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    source = images_01.mul(2.0).sub(1.0)
    domain = torch.full((len(source),), CAT_DOMAIN, device=source.device, dtype=torch.long)
    source_raw = style_encoder(source, domain)
    source_style = standardizer.transform(source_raw)
    target_style = lightsb.conditional_mean(source_style)
    target_raw = standardizer.inverse(target_style)
    target_image = renderer(source, target_raw).add(1.0).div(2.0).clamp(0.0, 1.0)
    source_feature, _, _ = dino_features_with_grad(dino, images_01, preprocess)
    target_feature, _, _ = dino_features_with_grad(dino, target_image, preprocess)
    context = torch.cat(
        [source_style, target_style - source_style, source_feature], dim=1
    )
    return source, source_raw, target_raw, context, source_feature, target_feature


def train_controller(
    controller: StyleController,
    renderer: Generator,
    style_encoder: torch.nn.Module,
    lightsb: DiagonalLightSB,
    standardizer: CoordinateStandardizer,
    dino: torch.nn.Module,
    species_direction: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    dataset = AFHQDomainDataset(
        args.afhq_root,
        "cat",
        image_size=256,
        training=True,
        augment=True,
        limit=args.train_limit,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    preprocess = BatchPreprocessor(64)
    times = torch.linspace(0.0, 1.0, args.train_times, device=device)
    interior_time = times[1:-1]
    history: list[dict[str, float]] = []
    direction = F.normalize(species_direction.to(device), dim=0)
    controller.train()
    for step in range(1, args.steps + 1):
        images, iterator = next_images(iterator, loader)
        images = images.to(device)
        with torch.no_grad():
            source, source_raw, target_raw, context, source_feature, target_feature = endpoint_state(
                images,
                style_encoder,
                lightsb,
                standardizer,
                renderer,
                dino,
                preprocess,
            )
        layer_styles, clock, raw_clock, residual = controlled_style_path(
            controller,
            source_raw,
            target_raw,
            context,
            times,
            clock_tube=args.clock_tube,
        )
        interior_styles = layer_styles[:, 1:-1]
        repeated_source = source[:, None].expand(
            -1, len(interior_time), -1, -1, -1
        ).reshape(-1, 3, 256, 256)
        rendered = renderer(
            repeated_source, interior_styles.reshape(-1, controller.layers, controller.style_dimension)
        ).add(1.0).div(2.0).clamp(0.0, 1.0)
        rendered_path = rendered.reshape(
            len(images), len(interior_time), 3, 256, 256
        )
        rendered_feature, _, _ = dino_features_with_grad(dino, rendered, preprocess)
        rendered_feature = rendered_feature.reshape(len(images), len(interior_time), -1)

        weight = interior_time[None, :, None]
        target_path = F.normalize(
            torch.lerp(source_feature[:, None], target_feature[:, None], weight), dim=2
        )
        geodesic = (1.0 - (rendered_feature * target_path).sum(2)).mean()

        source_coordinate = source_feature @ direction
        target_coordinate = target_feature @ direction
        desired_coordinate = torch.lerp(
            source_coordinate[:, None], target_coordinate[:, None], interior_time[None]
        )
        coordinate = rendered_feature @ direction
        semantic = F.smooth_l1_loss(coordinate, desired_coordinate)

        source_content = project_off_direction(source_feature, direction)
        rendered_content = project_off_direction(rendered_feature, direction)
        content = (1.0 - (rendered_content * source_content[:, None]).sum(2)).mean()

        full_features = torch.cat(
            [source_feature[:, None], rendered_feature, target_feature[:, None]], dim=1
        )
        speed = torch.linalg.vector_norm(full_features[:, 1:] - full_features[:, :-1], dim=2)
        normalized_speed = speed / speed.mean(1, keepdim=True).detach().clamp_min(1e-4)
        constant_speed = (normalized_speed - 1.0).square().mean()

        interior_tv = total_variation(rendered).reshape(len(images), len(interior_time))
        with torch.no_grad():
            target_images = renderer(source, target_raw).add(1.0).div(2.0).clamp(0.0, 1.0)
            endpoint_tv = torch.maximum(
                total_variation(images),
                total_variation(target_images),
            )
        artifact = torch.relu(interior_tv - args.tv_margin * endpoint_tv[:, None]).mean()
        chroma_artifact = localized_chroma_artifact_loss(
            rendered_path,
            torch.stack([images, target_images], dim=1),
            topk_fraction=args.artifact_topk_fraction,
            margin=args.chroma_margin,
        )
        support = layerwise_support_regularizer(raw_clock, args.clock_tube)
        residual_energy, residual_curvature = residual_style_regularizers(
            residual, source_raw, target_raw
        )
        clock_penalty = layerwise_clock_regularizer(clock, times)

        loss = (
            args.lambda_geodesic * geodesic
            + args.lambda_semantic * semantic
            + args.lambda_content * content
            + args.lambda_speed * constant_speed
            + args.lambda_artifact * artifact
            + args.lambda_chroma * chroma_artifact
            + args.lambda_support * support
            + args.lambda_residual * residual_energy
            + args.lambda_residual_curvature * residual_curvature
            + args.lambda_clock * clock_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "geodesic": float(geodesic.detach()),
                "semantic": float(semantic.detach()),
                "content": float(content.detach()),
                "constant_speed": float(constant_speed.detach()),
                "artifact": float(artifact.detach()),
                "chroma_artifact": float(chroma_artifact.detach()),
                "support": float(support.detach()),
                "residual_energy": float(residual_energy.detach()),
                "residual_curvature": float(residual_curvature.detach()),
                "clock": float(clock_penalty.detach()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


@torch.inference_mode()
def render_shared_style(
    renderer: Generator,
    sources: Tensor,
    styles: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    batch, time_count, dimension = styles.shape
    images = sources[:, None].expand(-1, time_count, -1, -1, -1).reshape(-1, 3, 256, 256)
    rendered = [
        renderer(image, style).add(1.0).div(2.0).clamp(0.0, 1.0).cpu()
        for image, style in zip(images.split(batch_size), styles.reshape(-1, dimension).split(batch_size))
    ]
    path = torch.cat(rendered).reshape(batch, time_count, 3, 256, 256)
    path[:, 0] = sources.cpu().add(1.0).div(2.0)
    return path


@torch.inference_mode()
def render_layerwise_style(
    renderer: Generator,
    sources: Tensor,
    styles: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    batch, time_count, layers, dimension = styles.shape
    images = sources[:, None].expand(-1, time_count, -1, -1, -1).reshape(-1, 3, 256, 256)
    rendered = [
        renderer(image, style).add(1.0).div(2.0).clamp(0.0, 1.0).cpu()
        for image, style in zip(
            images.split(batch_size), styles.reshape(-1, layers, dimension).split(batch_size)
        )
    ]
    path = torch.cat(rendered).reshape(batch, time_count, 3, 256, 256)
    path[:, 0] = sources.cpu().add(1.0).div(2.0)
    return path


def gradual_metrics(
    result: dict[str, object], features: Tensor, path: Tensor
) -> dict[str, float]:
    probability = torch.tensor(result["path_dog_probability"])
    probability_step = (probability[:, 1:] - probability[:, :-1]).abs()
    speed = torch.linalg.vector_norm(features[:, 1:] - features[:, :-1], dim=2)
    speed_cv = speed.std(1, unbiased=False) / speed.mean(1).clamp_min(1e-5)
    endpoint_tv = torch.maximum(total_variation(path[:, 0]), total_variation(path[:, -1]))
    interior_tv = torch.stack([total_variation(path[:, index]) for index in range(1, path.shape[1] - 1)], dim=1)
    return {
        "mean_max_probability_step": float(probability_step.max(1).values.mean()),
        "mean_dino_speed_cv_per_path": float(speed_cv.mean()),
        "mean_active_probability_steps": float((probability_step >= 0.03).float().sum(1).mean()),
        "mean_transition_frame_fraction": float(
            ((probability > 0.1) & (probability < 0.9)).float().mean()
        ),
        "mean_tv_excess_over_endpoints": float(
            torch.relu(interior_tv - endpoint_tv[:, None]).mean()
        ),
        "topk_chroma_laplacian_excess": float(
            localized_chroma_artifact_loss(
                path[:, 1:-1],
                path[:, [0, -1]],
                topk_fraction=0.01,
                margin=1.1,
            )
        ),
    }


@torch.inference_mode()
def evaluate(
    controller: StyleController,
    renderer: Generator,
    style_encoder: torch.nn.Module,
    lightsb: DiagonalLightSB,
    standardizer: CoordinateStandardizer,
    dino: torch.nn.Module,
    probe: BinaryLinearProbe,
    species_direction: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    test = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=256, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    generator = torch.Generator().manual_seed(args.seed + 5)
    cat_indices = torch.randperm(len(test["cat"]), generator=generator)[: args.eval_examples]
    dog_indices = torch.randperm(len(test["dog"]), generator=generator)[: args.eval_examples]
    cats = torch.stack([test["cat"][int(index)] for index in cat_indices])
    dogs = torch.stack([test["dog"][int(index)] for index in dog_indices])
    source = cats.to(device).mul(2.0).sub(1.0)
    preprocess = BatchPreprocessor(64)
    source_raw_rows: list[Tensor] = []
    target_raw_rows: list[Tensor] = []
    context_rows: list[Tensor] = []
    for images in cats.to(device).split(args.encode_batch_size):
        _, source_raw, target_raw, context, _, _ = endpoint_state(
            images, style_encoder, lightsb, standardizer, renderer, dino, preprocess
        )
        source_raw_rows.append(source_raw)
        target_raw_rows.append(target_raw)
        context_rows.append(context)
    source_raw = torch.cat(source_raw_rows)
    target_raw = torch.cat(target_raw_rows)
    context = torch.cat(context_rows)
    times = torch.linspace(0.0, 1.0, args.dense_steps, device=device)
    controlled_styles, clock, raw_clock, residual = controlled_style_path(
        controller,
        source_raw,
        target_raw,
        context,
        times,
        clock_tube=args.clock_tube,
    )
    clock_only_styles = controlled_styles - residual
    baseline_styles = torch.lerp(
        source_raw[:, None], target_raw[:, None], times[None, :, None]
    )
    paths: dict[str, Tensor] = {
        "global style chord": render_shared_style(
            renderer, source, baseline_styles, batch_size=args.render_batch_size
        ),
    }
    final_path_name = "learned layerwise physical-time path"
    if isinstance(controller, EndpointVanishingResidualStyleController):
        paths["same learned clocks without residual"] = render_layerwise_style(
            renderer, source, clock_only_styles, batch_size=args.render_batch_size
        )
        final_path_name = "learned endpoint-vanishing residual path"
    paths[final_path_name] = render_layerwise_style(
        renderer, source, controlled_styles, batch_size=args.render_batch_size
    )
    source_features, _, _ = dino_encode(dino, cats, preprocess, device)
    dog_features, _, _ = dino_encode(dino, dogs, preprocess, device)
    metrics: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        features = dino_path_features(dino, path, device=device)
        result = path_metrics(path, features, probe)
        result.update(gradual_metrics(result, features, path))
        result.update(
            endpoint_distribution_metrics(
                source_features, features[:, -1], dog_features, species_direction
            )
        )
        metrics[name] = result
        print(name + " " + json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}), flush=True)

    physical_indices = torch.linspace(
        0, args.dense_steps - 1, args.sheet_steps
    ).round().long().tolist()
    detail_indices = torch.linspace(
        0, args.dense_steps - 1, args.detail_sheet_steps
    ).round().long().tolist()
    unselected = slice(0, args.examples)
    displayed = {
        name: path[unselected][:, physical_indices] for name, path in paths.items()
    }
    save_source_only_sheet(
        displayed,
        args.output_dir / "uniform_physical_time_unselected.png",
        time_label="physical t",
    )
    save_source_only_sheet(
        {
            "learned residual path; fixed seeded test cats": paths[
                final_path_name
            ][unselected][:, detail_indices]
        },
        args.output_dir / "uniform_physical_time_layerwise_detail.png",
        time_label="physical t",
    )
    make_distribution_gif(
        pil_paths(paths[final_path_name][unselected]),
        args.output_dir / "uniform_physical_time_unselected.gif",
        clock_label="physical",
        duration_ms=args.frame_duration_ms,
    )
    save_probability_plot(metrics, args.output_dir / "physical_time_probability.png")
    torch.save(paths, args.output_dir / "physical_time_paths.pt")
    reference_endpoint = paths["global style chord"][:, -1]
    endpoint_l1 = max(
        float((path[:, -1] - reference_endpoint).abs().max())
        for path in paths.values()
    )
    residual_energy, residual_curvature = residual_style_regularizers(
        residual, source_raw, target_raw
    )
    return {
        "cat_test_indices": cat_indices.tolist(),
        "dog_evaluation_indices": dog_indices.tolist(),
        "unselected_demo_eval_rows": list(range(args.examples)),
        "physical_sheet_dense_indices": physical_indices,
        "detail_sheet_dense_indices": detail_indices,
        "endpoint_max_absolute_difference": endpoint_l1,
        "final_path_name": final_path_name,
        "clock_clip_fraction": float((clock != raw_clock).float().mean().cpu()),
        "raw_support_tube_penalty": float(
            layerwise_support_regularizer(raw_clock, args.clock_tube).cpu()
        ),
        "metrics": metrics,
        "residual_relative_energy": float(residual_energy.cpu()),
        "residual_temporal_curvature": float(residual_curvature.cpu()),
        "mean_layer_clocks": clock.mean(0).cpu().tolist(),
        "raw_mean_layer_clocks": raw_clock.mean(0).cpu().tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    device = torch.device("mps")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer, style_encoder = load_afhq_renderer(str(args.stargan_checkpoint), device=device)
    dino = load_dino(args, device)
    probe, species_direction, probe_metrics = fit_or_load_afhq_probe(dino, args, device)
    lightsb, standardizer = load_transport(args, device)
    controller_class = (
        EndpointVanishingResidualStyleController
        if args.residual_rank > 0
        else MonotoneLayerwiseStyleController
    )
    controller_kwargs: dict[str, object] = {
        "style_dimension": lightsb.dimension,
        "context_dimension": 2 * lightsb.dimension + species_direction.numel(),
        "layers": len(renderer.decode),
        "hidden_dimension": args.hidden_dimension,
    }
    if args.residual_rank > 0:
        controller_kwargs.update(
            residual_rank=args.residual_rank,
            max_residual_ratio=args.max_residual_ratio,
        )
    controller = controller_class(**controller_kwargs).to(device)
    if isinstance(controller, EndpointVanishingResidualStyleController):
        if args.base_controller_checkpoint is not None:
            base_payload = torch.load(
                args.base_controller_checkpoint, map_location="cpu", weights_only=False
            )
            controller.clock_controller.load_state_dict(base_payload["controller"])
        if not args.train_clock:
            controller.clock_controller.requires_grad_(False)
    checkpoint = args.output_dir / "layerwise_style_controller.pt"
    if args.evaluate_only:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        controller.load_state_dict(saved["controller"])
        history = saved["history"]
    else:
        history = train_controller(
            controller,
            renderer,
            style_encoder,
            lightsb,
            standardizer,
            dino,
            species_direction,
            args,
            device,
        )
        torch.save(
            {
                "controller": controller.state_dict(),
                "controller_class": controller_class.__name__,
                "history": history,
            },
            checkpoint,
        )
    controller.eval()
    evaluation = evaluate(
        controller,
        renderer,
        style_encoder,
        lightsb,
        standardizer,
        dino,
        probe,
        species_direction,
        args,
        device,
    )
    summary: dict[str, object] = {
        "claim": (
            "low-rank endpoint-vanishing residual around a frozen learned layerwise LightSB path"
            if isinstance(controller, EndpointVanishingResidualStyleController)
            else "shared layerwise observation controller for a frozen source-only LightSB endpoint map"
        ),
        "leakage_boundary": "Controller training uses AFHQ train cats only. Held-out cats are unselected and no dog image enters inference. Dogs are evaluation-only.",
        "display_clock": "Uniform physical time; no DINO semantic subsampling or per-path re-timing.",
        "endpoint": "Exactly unchanged: all block clocks equal zero at t=0 and one at t=1.",
        "support_control": "Layer clocks are projected into a tube around StarGAN's shared-style diagonal; a soft pre-projection barrier and localized top-k chroma-Laplacian loss discourage off-manifold color spikes.",
        "afhq_probe_test_metrics": probe_metrics,
        "evaluation": evaluation,
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument(
        "--stargan-checkpoint", type=Path,
        default=project / "data/stargan-v2/expr/checkpoints/afhq/100000_nets_ema.ckpt",
    )
    parser.add_argument("--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt")
    parser.add_argument(
        "--probe-cache", type=Path,
        default=project / "outputs/full_dataset_latent_bridge_demo/afhq_dino_linear_probe.pt",
    )
    parser.add_argument(
        "--transport-checkpoint", type=Path,
        default=project / "outputs/stargan_style_lightsb/style_transport.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project / "outputs/layerwise_style_lightsb_residual_frozen",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-times", type=int, default=7)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lambda-geodesic", type=float, default=1.0)
    parser.add_argument("--lambda-semantic", type=float, default=0.5)
    parser.add_argument("--lambda-content", type=float, default=0.5)
    parser.add_argument("--lambda-speed", type=float, default=0.25)
    parser.add_argument("--lambda-artifact", type=float, default=0.2)
    parser.add_argument("--lambda-chroma", type=float, default=2.0)
    parser.add_argument("--lambda-support", type=float, default=0.05)
    parser.add_argument("--lambda-residual", type=float, default=0.05)
    parser.add_argument("--lambda-residual-curvature", type=float, default=0.05)
    parser.add_argument("--lambda-clock", type=float, default=0.01)
    parser.add_argument("--tv-margin", type=float, default=1.1)
    parser.add_argument("--chroma-margin", type=float, default=1.1)
    parser.add_argument("--artifact-topk-fraction", type=float, default=0.01)
    parser.add_argument("--clock-tube", type=float, default=0.2)
    parser.add_argument("--residual-rank", type=int, default=8)
    parser.add_argument("--max-residual-ratio", type=float, default=0.35)
    parser.add_argument(
        "--base-controller-checkpoint",
        type=Path,
        default=project
        / "outputs/layerwise_style_lightsb_regularized/layerwise_style_controller.pt",
    )
    parser.add_argument(
        "--train-clock",
        action="store_true",
        help="jointly update layer clocks instead of freezing the loaded base controller",
    )
    parser.add_argument("--dense-steps", type=int, default=25)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--detail-sheet-steps", type=int, default=13)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--frame-duration-ms", type=int, default=140)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if (
        args.train_times < 4
        or args.dense_steps < max(args.sheet_steps, args.detail_sheet_steps)
        or min(args.sheet_steps, args.detail_sheet_steps) < 2
    ):
        parser.error("invalid time-grid sizes")
    if not 1 <= args.examples <= args.eval_examples:
        parser.error("examples must be positive and no larger than eval examples")
    if not 0 < args.artifact_topk_fraction <= 1:
        parser.error("artifact-topk-fraction must lie in (0, 1]")
    if not 0 <= args.clock_tube <= 1:
        parser.error("clock-tube must lie in [0, 1]")
    if args.residual_rank < 0:
        parser.error("residual-rank must be nonnegative")
    if args.max_residual_ratio <= 0:
        parser.error("max-residual-ratio must be positive")
    return args


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(run(configuration), indent=2))
