"""Train the 32x32 latent bridge objective on all non-quarantined AFHQ data."""

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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.features import (
    compact_spatial_content_descriptor,
    content_cost_matrix,
    species_direction,
)
from brown_bridge.latent_image_bridge import LatentImageAutoencoder
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.ot import sinkhorn_from_cost
from render_dino_stargan_bridge import (
    DISPLAY_STEPS,
    dino_encode,
    make_contact_sheet,
    make_distribution_gif,
    tensor_to_pil,
    unique_minimum_assignment,
)
from train_continuous_vae_renderer import AFHQDomainDataset, image_edges


class IndexedDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, dataset: AFHQDomainDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.dataset[index], index


def load_dino(args: argparse.Namespace, device: torch.device) -> DinoCatDogClassifier:
    model = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb", image_size=64, pretrained=False, fusion="gated"
    )
    model.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def extract_raw_features(
    dino: DinoCatDogClassifier,
    dataset: AFHQDomainDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    preprocess = BatchPreprocessor(64)
    loader = DataLoader(dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=0)
    global_batches: list[Tensor] = []
    patch_batches: list[Tensor] = []
    for batch_index, images in enumerate(loader, start=1):
        normalized = preprocess(images.to(device), training=False)
        global_state, _, patches = dino.encode_spatial(normalized)
        if patches is None:
            raise RuntimeError("DINO backbone did not expose patch tokens")
        global_batches.append(F.normalize(global_state.float(), dim=1).cpu())
        patch_batches.append(F.normalize(patches.float(), dim=2).cpu())
        if batch_index % 50 == 0:
            print(f"features {batch_index * args.feature_batch_size}/{len(dataset)}", flush=True)
    return torch.cat(global_batches), torch.cat(patch_batches)


@torch.no_grad()
def build_descriptor_cache(
    dino: DinoCatDogClassifier,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Tensor | list[str]], dict[str, AFHQDomainDataset]]:
    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root,
            domain,
            image_size=128,
            training=True,
            augment=False,
            limit=args.train_limit,
        )
        for domain in ("cat", "dog")
    }
    names = {domain: [path.name for path in dataset.paths] for domain, dataset in datasets.items()}
    if args.descriptor_cache.exists():
        cached = torch.load(args.descriptor_cache, map_location="cpu", weights_only=False)
        if cached.get("cat_names") == names["cat"] and cached.get("dog_names") == names["dog"]:
            return cached, datasets
        raise RuntimeError("descriptor cache paths do not match the filtered AFHQ datasets")
    cat_global, cat_patch = extract_raw_features(dino, datasets["cat"], args, device)
    dog_global, dog_patch = extract_raw_features(dino, datasets["dog"], args, device)
    direction = species_direction(cat_global, dog_global)
    cache: dict[str, Tensor | list[str]] = {
        "cat_descriptor": compact_spatial_content_descriptor(cat_global, cat_patch, direction),
        "dog_descriptor": compact_spatial_content_descriptor(dog_global, dog_patch, direction),
        "species_direction": direction,
        "cat_names": names["cat"],
        "dog_names": names["dog"],
    }
    torch.save(cache, args.descriptor_cache)
    return cache, datasets


def reconstruction_objective(prediction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    pixel = F.l1_loss(prediction, target)
    pred_dx, pred_dy = image_edges(prediction)
    target_dx, target_dy = image_edges(target)
    edge = 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))
    pyramid = 0.5 * (
        F.l1_loss(F.avg_pool2d(prediction, 2), F.avg_pool2d(target, 2))
        + F.l1_loss(F.avg_pool2d(prediction, 4), F.avg_pool2d(target, 4))
    )
    return pixel + 0.25 * edge + 0.1 * pyramid, {
        "pixel": pixel,
        "edge": edge,
        "pyramid": pyramid,
    }


def next_batch(
    iterator: object, loader: DataLoader
) -> tuple[tuple[Tensor, Tensor], object]:
    try:
        batch = next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def train(
    cache: dict[str, Tensor | list[str]],
    deterministic: dict[str, AFHQDomainDataset],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[LatentImageAutoencoder, list[dict[str, float]]]:
    augmented = {
        domain: AFHQDomainDataset(
            args.afhq_root,
            domain,
            image_size=128,
            training=True,
            augment=True,
            limit=args.train_limit,
        )
        for domain in ("cat", "dog")
    }
    for domain in ("cat", "dog"):
        if [path.name for path in augmented[domain].paths] != [
            path.name for path in deterministic[domain].paths
        ]:
            raise RuntimeError("augmented and deterministic dataset indices differ")
    loaders = {
        domain: DataLoader(
            IndexedDataset(dataset),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        for domain, dataset in augmented.items()
    }
    iterators = {domain: iter(loader) for domain, loader in loaders.items()}
    cat_descriptor = cache["cat_descriptor"]
    dog_descriptor = cache["dog_descriptor"]
    assert isinstance(cat_descriptor, Tensor) and isinstance(dog_descriptor, Tensor)
    model = LatentImageAutoencoder(args.latent_channels, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator().manual_seed(args.seed + 41)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, args.steps + 1):
        (cat_images, cat_indices), iterators["cat"] = next_batch(
            iterators["cat"], loaders["cat"]
        )
        (dog_images, dog_indices), iterators["dog"] = next_batch(
            iterators["dog"], loaders["dog"]
        )
        cost = content_cost_matrix(cat_descriptor[cat_indices], dog_descriptor[dog_indices])
        coupling = sinkhorn_from_cost(
            cost,
            epsilon=args.ot_epsilon,
            max_iterations=200,
            tolerance=1e-6,
        )
        flat = torch.multinomial(
            coupling.plan.flatten(), args.batch_size, replacement=True, generator=generator
        )
        source_index = torch.div(flat, args.batch_size, rounding_mode="floor")
        target_index = flat.remainder(args.batch_size)
        cat_images = cat_images.to(device)
        dog_images = dog_images.to(device)
        endpoints = torch.cat([cat_images, dog_images])
        latent = model.encode(endpoints)
        reconstruction = model.decode(latent)
        endpoint_loss, endpoint_pieces = reconstruction_objective(reconstruction, endpoints)
        cat_latent, dog_latent = latent.chunk(2)
        left = cat_latent[source_index.to(device)]
        right = dog_latent[target_index.to(device)]
        time = 0.08 + 0.84 * torch.rand(args.batch_size, generator=generator).to(device)
        delta = args.acceleration_delta
        times = torch.stack([time - delta, time, time + delta]).clamp(0.0, 1.0)
        bridge_latent = (
            (1.0 - times[:, :, None, None, None]) * left[None]
            + times[:, :, None, None, None] * right[None]
        )
        if args.diffusivity > 0:
            noise = torch.randn(left.shape, generator=generator).to(device)
            scale = torch.sqrt(2.0 * args.diffusivity * time * (1.0 - time))
            bridge_latent[1] = bridge_latent[1] + scale[:, None, None, None] * noise
        bridge_images = model.decode(bridge_latent.flatten(0, 1)).reshape(
            3, args.batch_size, 3, 128, 128
        )
        center = bridge_images[1]
        latent_cycle = F.l1_loss(model.encode(center), bridge_latent[1].detach())
        acceleration = (bridge_images[2] - 2.0 * center + bridge_images[0]).abs().mean()
        path_ramp = min(1.0, step / args.path_warmup)
        loss = endpoint_loss + path_ramp * (
            args.lambda_latent_cycle * latent_cycle
            + args.lambda_acceleration * acceleration
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            expected_cost = float((coupling.plan * cost).sum())
            random_cost = float(cost.mean())
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "pixel": float(endpoint_pieces["pixel"].detach()),
                "edge": float(endpoint_pieces["edge"].detach()),
                "latent_cycle": float(latent_cycle.detach()),
                "acceleration": float(acceleration.detach()),
                "ot_expected_cost": expected_cost,
                "random_expected_cost": random_cost,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    return model.eval(), history


@torch.no_grad()
def evaluate_reconstruction(
    model: LatentImageAutoencoder,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    test = torch.utils.data.ConcatDataset(
        [
            AFHQDomainDataset(
                args.afhq_root, domain, image_size=128, training=False, augment=False
            )
            for domain in ("cat", "dog")
        ]
    )
    loader = DataLoader(test, batch_size=args.validation_batch, shuffle=False)
    total_l1 = total_edge = 0.0
    count = 0
    for batch_index, images in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        images = images.to(device)
        reconstruction, _ = model(images)
        pred_dx, pred_dy = image_edges(reconstruction)
        target_dx, target_dy = image_edges(images)
        total_l1 += len(images) * float(F.l1_loss(reconstruction, images))
        total_edge += len(images) * float(
            0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))
        )
        count += len(images)
    return {"l1": total_l1 / count, "edge_l1": total_edge / count}


@torch.no_grad()
def render_test_pairs(
    model: LatentImageAutoencoder,
    dino: DinoCatDogClassifier,
    cache: dict[str, Tensor | list[str]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    cats = AFHQDomainDataset(args.afhq_root, "cat", image_size=128, training=False, augment=False)
    dogs = AFHQDomainDataset(args.afhq_root, "dog", image_size=128, training=False, augment=False)
    cat_tensor = torch.stack([cats[index] for index in range(5)])
    dog_tensor = torch.stack([dogs[index] for index in range(8)])
    preprocess = BatchPreprocessor(64)
    cat_global, cat_patch, _ = dino_encode(dino, cat_tensor, preprocess, device)
    dog_global, dog_patch, _ = dino_encode(dino, dog_tensor, preprocess, device)
    direction = cache["species_direction"]
    assert isinstance(direction, Tensor)
    cat_descriptor = compact_spatial_content_descriptor(cat_global, cat_patch, direction)
    dog_descriptor = compact_spatial_content_descriptor(dog_global, dog_patch, direction)
    assignment = unique_minimum_assignment(content_cost_matrix(cat_descriptor, dog_descriptor))
    paired_dogs = dog_tensor[assignment]
    endpoints = torch.cat([cat_tensor, paired_dogs]).to(device)
    latent = model.encode(endpoints)
    cat_latent, dog_latent = latent.chunk(2)
    time = torch.linspace(0.0, 1.0, DISPLAY_STEPS, device=device)
    path_latent = (
        (1.0 - time[:, None, None, None, None]) * cat_latent[None]
        + time[:, None, None, None, None] * dog_latent[None]
    )
    decoded = model.decode(path_latent.flatten(0, 1)).reshape(
        DISPLAY_STEPS, 5, 3, 128, 128
    ).transpose(0, 1)
    path_pil = [
        [
            tensor_to_pil(image.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
            for image in row
        ]
        for row in decoded.cpu()
    ]
    cat_pil = [
        tensor_to_pil(image.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
        for image in cat_tensor
    ]
    dog_pil = [
        tensor_to_pil(image.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
        for image in paired_dogs
    ]
    make_contact_sheet(
        cat_pil,
        path_pil,
        dog_pil,
        args.output_dir / "full_dataset_test_bridge.png",
        clock_label="full-data latent",
    )
    make_distribution_gif(
        path_pil,
        args.output_dir / "full_dataset_test_bridge.gif",
        clock_label="full-data latent",
    )
    features, _, probability = dino_encode(
        dino, decoded.flatten(0, 1).cpu(), preprocess, device
    )
    probability = probability.reshape(5, DISPLAY_STEPS)
    features = features.reshape(5, DISPLAY_STEPS, -1)
    coordinate = features @ direction.to(features)
    adjacent = (decoded[:, 1:] - decoded[:, :-1]).abs().mean()
    torch.save(
        {"trajectory_128": decoded.cpu(), "paired_dog_indices": assignment},
        args.output_dir / "full_dataset_test_trajectories.pt",
    )
    return {
        "paired_dog_indices": assignment,
        "dog_probability": probability.tolist(),
        "species_coordinate": coordinate.tolist(),
        "mean_adjacent_l1": float(adjacent),
    }


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--descriptor-cache", type=Path)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--validation-batch", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ot-epsilon", type=float, default=0.10)
    parser.add_argument("--diffusivity", type=float, default=0.0005)
    parser.add_argument("--lambda-latent-cycle", type=float, default=0.05)
    parser.add_argument("--lambda-acceleration", type=float, default=0.02)
    parser.add_argument("--acceleration-delta", type=float, default=0.05)
    parser.add_argument("--path-warmup", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.descriptor_cache is None:
        args.descriptor_cache = args.output_dir / "full_afhq_dino_descriptors.pt"
    return args


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("mps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dino = load_dino(args, device)
    cache, deterministic = build_descriptor_cache(dino, args, device)
    model, history = train(cache, deterministic, args, device)
    validation = evaluate_reconstruction(model, args, device)
    rendering = render_test_pairs(model, dino, cache, args, device)
    cat_names = cache["cat_names"]
    dog_names = cache["dog_names"]
    assert isinstance(cat_names, list) and isinstance(dog_names, list)
    quarantine = {
        "cat_excluded": "flickr_cat_000002.png" not in cat_names,
        "dog_excluded": "flickr_dog_000002.png" not in dog_names,
    }
    summary = {
        "status": "full_non_quarantined_dataset_training",
        "training_cats": len(cat_names),
        "training_dogs": len(dog_names),
        "quarantine_verified": quarantine,
        "validation": validation,
        "render": rendering,
        "history": history,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(
        {"model": model.cpu().state_dict(), "summary": summary},
        args.output_dir / "full_dataset_latent_bridge.pt",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
