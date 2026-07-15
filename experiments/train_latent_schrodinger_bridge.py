"""Train a 32x32 AFHQ image latent and reciprocal cat-to-dog bridge on MPS."""

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
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.features import content_cost_matrix, content_descriptor, species_direction
from brown_bridge.latent_image_bridge import (
    LatentImageAutoencoder,
    SpatialReciprocalField,
    integrate_spatial_probability_flow,
    integrate_spatial_sde,
    make_spatial_reciprocal_batch,
)
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.ot import sinkhorn_from_cost
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
from train_continuous_vae_renderer import AFHQDomainDataset, image_edges


def reconstruction_loss(
    reconstruction: Tensor,
    image: Tensor,
    latent: Tensor,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    pixel = F.l1_loss(reconstruction, image)
    pred_dx, pred_dy = image_edges(reconstruction)
    image_dx, image_dy = image_edges(image)
    edge = 0.5 * (F.l1_loss(pred_dx, image_dx) + F.l1_loss(pred_dy, image_dy))
    pyramid = 0.5 * (
        F.l1_loss(F.avg_pool2d(reconstruction, 2), F.avg_pool2d(image, 2))
        + F.l1_loss(F.avg_pool2d(reconstruction, 4), F.avg_pool2d(image, 4))
    )
    channel_mean = latent.mean((0, 2, 3))
    channel_std = latent.std((0, 2, 3))
    moment = channel_mean.square().mean() + F.relu(0.2 - channel_std).square().mean()
    total = (
        pixel
        + args.lambda_edge * edge
        + args.lambda_pyramid * pyramid
        + args.lambda_latent_moment * moment
    )
    return total, {"pixel": pixel, "edge": edge, "pyramid": pyramid, "moment": moment}


def datasets(args: argparse.Namespace) -> tuple[ConcatDataset[Tensor], ConcatDataset[Tensor]]:
    train = ConcatDataset(
        [
            AFHQDomainDataset(args.afhq_root, domain, image_size=128, training=True)
            for domain in ("cat", "dog")
        ]
    )
    test = ConcatDataset(
        [
            AFHQDomainDataset(args.afhq_root, domain, image_size=128, training=False)
            for domain in ("cat", "dog")
        ]
    )
    return train, test


def train_autoencoder(
    args: argparse.Namespace, device: torch.device
) -> tuple[LatentImageAutoencoder, list[dict[str, float]]]:
    train_data, _ = datasets(args)
    loader = DataLoader(
        train_data, batch_size=args.ae_batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    iterator = iter(loader)
    model = LatentImageAutoencoder(args.latent_channels, args.ae_base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ae_learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ae_steps)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, args.ae_steps + 1):
        try:
            image = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image = next(iterator)
        image = image.to(device)
        reconstruction, latent = model(image)
        loss, pieces = reconstruction_loss(reconstruction, image, latent, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.ae_steps:
            record = {"step": float(step), "loss": float(loss.detach())}
            record.update({key: float(value.detach()) for key, value in pieces.items()})
            history.append(record)
            print("AE " + json.dumps(record), flush=True)
    return model.eval(), history


@torch.no_grad()
def evaluate_autoencoder(
    model: LatentImageAutoencoder,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    _, test_data = datasets(args)
    loader = DataLoader(test_data, batch_size=args.validation_batch, shuffle=False)
    totals = {"l1": 0.0, "edge_l1": 0.0, "edge_energy": 0.0, "real_edge_energy": 0.0}
    count = 0
    for batch_index, image in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        image = image.to(device)
        reconstruction, _ = model(image)
        pred_dx, pred_dy = image_edges(reconstruction)
        image_dx, image_dy = image_edges(image)
        batch = len(image)
        totals["l1"] += batch * float(F.l1_loss(reconstruction, image))
        totals["edge_l1"] += batch * float(
            0.5 * (F.l1_loss(pred_dx, image_dx) + F.l1_loss(pred_dy, image_dy))
        )
        totals["edge_energy"] += batch * float(
            0.5 * (pred_dx.abs().mean() + pred_dy.abs().mean())
        )
        totals["real_edge_energy"] += batch * float(
            0.5 * (image_dx.abs().mean() + image_dy.abs().mean())
        )
        count += batch
    return {key: value / count for key, value in totals.items()}


def load_dino(args: argparse.Namespace, device: torch.device) -> DinoCatDogClassifier:
    model = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb",
        image_size=64,
        pretrained=False,
        fusion="gated",
    )
    model.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def encode_endpoint_pool(
    autoencoder: LatentImageAutoencoder,
    dino: DinoCatDogClassifier,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Tensor]:
    preprocess = BatchPreprocessor(64)
    result: dict[str, Tensor] = {}
    for domain in ("cat", "dog"):
        dataset = AFHQDomainDataset(
            args.afhq_root,
            domain,
            image_size=128,
            training=True,
            limit=args.pool_size,
        )
        loader = DataLoader(dataset, batch_size=args.pool_batch_size, shuffle=False)
        latents: list[Tensor] = []
        globals_: list[Tensor] = []
        patches_: list[Tensor] = []
        for image in loader:
            image = image.to(device)
            latents.append(autoencoder.encode(image).cpu())
            normalized = preprocess(image, training=False)
            global_state, _, patch_state = dino.encode_spatial(normalized)
            if patch_state is None:
                raise RuntimeError("DINO backbone did not expose patch tokens")
            globals_.append(F.normalize(global_state.float(), dim=1).cpu())
            patches_.append(F.normalize(patch_state.float(), dim=2).cpu())
        result[f"{domain}_latent"] = torch.cat(latents)
        result[f"{domain}_global"] = torch.cat(globals_)
        result[f"{domain}_patch"] = torch.cat(patches_)
    direction = species_direction(result["cat_global"], result["dog_global"])
    cat_descriptor = content_descriptor(result["cat_global"], result["cat_patch"], direction)
    dog_descriptor = content_descriptor(result["dog_global"], result["dog_patch"], direction)
    cost = content_cost_matrix(cat_descriptor, dog_descriptor)
    coupling = sinkhorn_from_cost(
        cost,
        epsilon=args.ot_epsilon,
        max_iterations=2_000,
        tolerance=1e-8,
    )
    result["species_direction"] = direction
    result["content_cost"] = cost.float()
    result["plan"] = coupling.plan.float()
    result["row_error"] = coupling.row_marginal_error.float()
    result["column_error"] = coupling.column_marginal_error.float()
    return result


def train_bridge(
    pool: dict[str, Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SpatialReciprocalField, list[dict[str, float]]]:
    field = SpatialReciprocalField(
        args.latent_channels,
        width=args.field_width,
        depth=args.field_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(field.parameters(), lr=args.field_learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.field_steps)
    generator = torch.Generator().manual_seed(args.seed + 301)
    probabilities = pool["plan"].double().flatten()
    cat_latent = pool["cat_latent"]
    dog_latent = pool["dog_latent"]
    history: list[dict[str, float]] = []
    field.train()
    for step in range(1, args.field_steps + 1):
        flat = torch.multinomial(
            probabilities, args.field_batch_size, replacement=True, generator=generator
        )
        source_index = torch.div(flat, len(dog_latent), rounding_mode="floor")
        target_index = flat.remainder(len(dog_latent))
        reciprocal = make_spatial_reciprocal_batch(
            cat_latent[source_index],
            dog_latent[target_index],
            diffusivity=args.diffusivity,
            generator=generator,
        )
        state = reciprocal.state.to(device)
        time = reciprocal.time.to(device)
        direction = reciprocal.direction.to(device)
        target_velocity = reciprocal.conditional_velocity.to(device)
        scaled_score = reciprocal.scaled_score.to(device)
        gamma = reciprocal.gamma.to(device)
        weight = reciprocal.flow_weight.to(device)
        velocity, score = field(state, time, direction)
        flow_per_sample = (velocity - target_velocity).square().mean((1, 2, 3))
        flow = (weight * flow_per_sample).sum() / weight.sum()
        score_loss = F.mse_loss(gamma[:, None, None, None] * score, scaled_score)
        reverse_velocity, reverse_score = field(state, 1.0 - time, -direction)
        reciprocal_loss = F.mse_loss(velocity, -reverse_velocity) + 0.1 * F.mse_loss(
            gamma[:, None, None, None] * score,
            gamma[:, None, None, None] * reverse_score,
        )
        dx = velocity[:, :, :, 1:] - velocity[:, :, :, :-1]
        dy = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]
        spatial = 0.5 * (dx.square().mean() + dy.square().mean())
        loss = (
            flow
            + score_loss
            + args.lambda_reciprocal * reciprocal_loss
            + args.lambda_spatial * spatial
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.field_steps:
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "flow": float(flow.detach()),
                "score": float(score_loss.detach()),
                "reciprocal": float(reciprocal_loss.detach()),
                "spatial": float(spatial.detach()),
            }
            history.append(record)
            print("SB " + json.dumps(record), flush=True)
    return field.eval(), history


def make_two_way_comparison(
    deterministic: list[list[Image.Image]],
    stochastic: list[list[Image.Image]],
    output: Path,
    *,
    deterministic_label: str = "probability flow",
    stochastic_label: str = "latent SDE",
) -> None:
    tile, header, label_width = 192, 34, 130
    rows, steps = len(deterministic), len(deterministic[0])
    canvas = Image.new("RGB", (label_width + steps * tile, header + 2 * rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for step in range(steps):
        draw.text((label_width + step * tile + 8, 11), f"t={step/(steps-1):.2f}", fill="black")
    for row in range(rows):
        for variant, (name, paths) in enumerate(
            ((deterministic_label, deterministic), (stochastic_label, stochastic))
        ):
            y = header + (2 * row + variant) * tile
            draw.text((5, y + 84), name, fill="black")
            for step, image in enumerate(paths[row]):
                canvas.paste(
                    image.resize((tile, tile), Image.Resampling.LANCZOS),
                    (label_width + step * tile, y),
                )
    canvas.save(output)


@torch.no_grad()
def render(
    autoencoder: LatentImageAutoencoder,
    field: SpatialReciprocalField,
    dino: DinoCatDogClassifier,
    pool: dict[str, Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    source_paths = sorted((args.assets / "src/cat").glob("*.jpg"))[:5]
    reference_paths = sorted((args.assets / "ref/dog").glob("*.jpg"))
    sources = [load_square(path) for path in source_paths]
    dogs = [load_square(path) for path in reference_paths]
    source_tensor = image_batch(sources)
    dog_tensor = image_batch(dogs)
    preprocess = BatchPreprocessor(64)
    cat_global, cat_patch, _ = dino_encode(dino, source_tensor, preprocess, device)
    dog_global, dog_patch, _ = dino_encode(dino, dog_tensor, preprocess, device)
    direction = species_direction(pool["cat_global"], pool["dog_global"])
    assignments = unique_minimum_assignment(
        content_cost_matrix(
            content_descriptor(cat_global, cat_patch, direction),
            content_descriptor(dog_global, dog_patch, direction),
        )
    )
    paired_dogs = [dogs[index] for index in assignments]
    initial = autoencoder.encode(
        F.interpolate(source_tensor, size=(128, 128), mode="bilinear", align_corners=False).to(device)
    )
    dense_flow = integrate_spatial_probability_flow(
        field, initial, direction=1, steps=args.integration_steps
    )
    generator = torch.Generator().manual_seed(args.seed + 907)
    noise = torch.randn((args.integration_steps, *initial.shape), generator=generator).to(device)
    dense_sde = integrate_spatial_sde(
        field,
        initial,
        diffusivity=args.sampling_diffusivity,
        direction=1,
        steps=args.integration_steps,
        noise=noise,
    )
    indices = torch.linspace(0, args.integration_steps, DISPLAY_STEPS).round().long()

    def decode(dense: Tensor) -> tuple[Tensor, list[list[Image.Image]]]:
        selected = dense[indices].transpose(0, 1)
        decoded = autoencoder.decode(selected.flatten(0, 1)).reshape(
            len(sources), DISPLAY_STEPS, 3, 128, 128
        )
        decoded[:, 0] = F.interpolate(
            source_tensor, size=(128, 128), mode="bilinear", align_corners=False
        ).to(device)
        pil = [
            [
                tensor_to_pil(image.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
                for image in row
            ]
            for row in decoded.cpu()
        ]
        return decoded, pil

    flow_images, flow_pil = decode(dense_flow)
    sde_images, sde_pil = decode(dense_sde)
    make_contact_sheet(
        sources,
        flow_pil,
        paired_dogs,
        args.output_dir / "latent_probability_flow.png",
        clock_label="latent SB",
    )
    make_distribution_gif(
        flow_pil,
        args.output_dir / "latent_probability_flow.gif",
        clock_label="latent SB",
    )
    make_two_way_comparison(
        flow_pil, sde_pil, args.output_dir / "probability_flow_vs_latent_sde.png"
    )
    all_images = torch.cat([flow_images.flatten(0, 1), sde_images.flatten(0, 1)])
    global_state, _, probability = dino_encode(dino, all_images.cpu(), preprocess, device)
    global_state = global_state.reshape(2, len(sources), DISPLAY_STEPS, -1)
    probability = probability.reshape(2, len(sources), DISPLAY_STEPS)
    direction = direction.to(global_state)
    coordinate = global_state @ direction
    flow_dx, flow_dy = image_edges(flow_images[:, 1:-1].flatten(0, 1))
    sde_dx, sde_dy = image_edges(sde_images[:, 1:-1].flatten(0, 1))
    torch.save(
        {
            "probability_flow_128": flow_images.cpu(),
            "latent_sde_128": sde_images.cpu(),
            "probability_flow_latent": dense_flow.cpu(),
            "latent_sde_latent": dense_sde.cpu(),
        },
        args.output_dir / "latent_bridge_trajectories.pt",
    )
    return {
        "probability_flow_dog_probability": probability[0].tolist(),
        "latent_sde_dog_probability": probability[1].tolist(),
        "probability_flow_species_coordinate": coordinate[0].tolist(),
        "latent_sde_species_coordinate": coordinate[1].tolist(),
        "probability_flow_interior_edge_energy": float(
            0.5 * (flow_dx.abs().mean() + flow_dy.abs().mean())
        ),
        "latent_sde_interior_edge_energy": float(
            0.5 * (sde_dx.abs().mean() + sde_dy.abs().mean())
        ),
        "probability_flow_adjacent_l1": float(
            (flow_images[:, 1:] - flow_images[:, :-1]).abs().mean()
        ),
        "latent_sde_adjacent_l1": float(
            (sde_images[:, 1:] - sde_images[:, :-1]).abs().mean()
        ),
        "paired_dog_indices": assignments,
    }


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument(
        "--assets", type=Path, default=project / "data/stargan-v2/assets/representative/afhq"
    )
    parser.add_argument("--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ae-steps", type=int, default=2000)
    parser.add_argument("--field-steps", type=int, default=2000)
    parser.add_argument("--ae-batch-size", type=int, default=12)
    parser.add_argument("--field-batch-size", type=int, default=16)
    parser.add_argument("--pool-batch-size", type=int, default=24)
    parser.add_argument("--validation-batch", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--ae-base-channels", type=int, default=32)
    parser.add_argument("--field-width", type=int, default=64)
    parser.add_argument("--field-depth", type=int, default=5)
    parser.add_argument("--ae-learning-rate", type=float, default=2e-4)
    parser.add_argument("--field-learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-edge", type=float, default=0.25)
    parser.add_argument("--lambda-pyramid", type=float, default=0.10)
    parser.add_argument("--lambda-latent-moment", type=float, default=1e-4)
    parser.add_argument("--lambda-reciprocal", type=float, default=0.05)
    parser.add_argument("--lambda-spatial", type=float, default=0.01)
    parser.add_argument("--ot-epsilon", type=float, default=0.10)
    parser.add_argument("--diffusivity", type=float, default=0.0025)
    parser.add_argument("--sampling-diffusivity", type=float, default=0.0005)
    parser.add_argument("--integration-steps", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("mps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    autoencoder, ae_history = train_autoencoder(args, device)
    ae_validation = evaluate_autoencoder(autoencoder, args, device)
    torch.save(
        {"model": autoencoder.cpu().state_dict(), "history": ae_history, "validation": ae_validation},
        args.output_dir / "latent_autoencoder.pt",
    )
    autoencoder.to(device)
    dino = load_dino(args, device)
    pool = encode_endpoint_pool(autoencoder, dino, args, device)
    torch.save(pool, args.output_dir / "afhq_endpoint_pool.pt")
    field, field_history = train_bridge(pool, args, device)
    render_metrics = render(autoencoder, field, dino, pool, args, device)
    summary = {
        "method": "32x32 latent empirical Schrodinger bridge with reciprocal flow/score field",
        "autoencoder_parameters": sum(p.numel() for p in autoencoder.parameters()),
        "field_parameters": sum(p.numel() for p in field.parameters()),
        "pool_size_per_domain": args.pool_size,
        "sinkhorn_row_error": float(pool["row_error"]),
        "sinkhorn_column_error": float(pool["column_error"]),
        "autoencoder_validation": ae_validation,
        "autoencoder_history": ae_history,
        "field_history": field_history,
        "render": render_metrics,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(
        {"field": field.cpu().state_dict(), "summary": summary},
        args.output_dir / "latent_schrodinger_bridge.pt",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
