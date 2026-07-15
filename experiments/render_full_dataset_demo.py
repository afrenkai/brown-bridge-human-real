"""Render an honest held-out demo of the full AFHQ latent-chord model.

The renderer deliberately distinguishes the demonstrated object—a decoded
latent chord between DINO-OT-selected endpoints—from a learned SDE.  An
independent AFHQ linear probe replaces the out-of-domain CIFAR classifier for
all semantic path diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.features import (
    compact_spatial_content_descriptor,
    content_cost_matrix,
    species_direction,
)
from brown_bridge.latent_image_bridge import LatentImageAutoencoder
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.probes import (
    BinaryLinearProbe,
    binary_probe_metrics,
    fit_binary_linear_probe,
)
from render_dino_stargan_bridge import (
    dino_encode,
    make_contact_sheet,
    make_distribution_gif,
    tensor_to_pil,
    unique_minimum_assignment,
)
from train_continuous_vae_renderer import AFHQDomainDataset


def dataset_fingerprint(dataset: AFHQDomainDataset) -> str:
    digest = hashlib.sha256()
    for path in dataset.paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
    return digest.hexdigest()


@torch.inference_mode()
def collect_global_features(
    dino: DinoCatDogClassifier,
    dataset: AFHQDomainDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    preprocess = BatchPreprocessor(64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches: list[Tensor] = []
    for batch_index, images in enumerate(loader, start=1):
        normalized = preprocess(images.to(device), training=False)
        features, _ = dino.encode(normalized)
        batches.append(F.normalize(features.float(), dim=1).cpu())
        if batch_index % 50 == 0:
            print(f"probe features {min(batch_index * batch_size, len(dataset))}/{len(dataset)}")
    return torch.cat(batches)


def fit_or_load_afhq_probe(
    dino: DinoCatDogClassifier,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BinaryLinearProbe, Tensor, dict[str, float]]:
    train = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=True, augment=False
        )
        for domain in ("cat", "dog")
    }
    test = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    fingerprints = {
        f"train_{domain}": dataset_fingerprint(train[domain]) for domain in ("cat", "dog")
    } | {f"test_{domain}": dataset_fingerprint(test[domain]) for domain in ("cat", "dog")}
    if args.probe_cache.exists():
        cached = torch.load(args.probe_cache, map_location="cpu", weights_only=False)
        if cached.get("fingerprints") != fingerprints:
            raise RuntimeError("AFHQ probe cache does not match the filtered datasets")
        return (
            BinaryLinearProbe.from_state_dict(cached["probe"]),
            cached["species_direction"].float(),
            cached["test_metrics"],
        )

    train_cat = collect_global_features(
        dino, train["cat"], batch_size=args.feature_batch_size, device=device
    )
    train_dog = collect_global_features(
        dino, train["dog"], batch_size=args.feature_batch_size, device=device
    )
    test_cat = collect_global_features(
        dino, test["cat"], batch_size=args.feature_batch_size, device=device
    )
    test_dog = collect_global_features(
        dino, test["dog"], batch_size=args.feature_batch_size, device=device
    )
    probe = fit_binary_linear_probe(train_cat, train_dog)
    metrics = binary_probe_metrics(probe, test_cat, test_dog)
    direction = species_direction(train_cat, train_dog)
    torch.save(
        {
            "probe": probe.state_dict(),
            "species_direction": direction,
            "test_metrics": metrics,
            "fingerprints": fingerprints,
            "train_counts": {domain: len(train[domain]) for domain in ("cat", "dog")},
            "test_counts": {domain: len(test[domain]) for domain in ("cat", "dog")},
        },
        args.probe_cache,
    )
    return probe, direction, metrics


def load_models(
    args: argparse.Namespace, device: torch.device
) -> tuple[LatentImageAutoencoder, DinoCatDogClassifier, dict[str, object]]:
    dino = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb", image_size=64, pretrained=False, fusion="gated"
    )
    dino.load_state_dict(
        torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True)
    )
    dino = dino.eval().requires_grad_(False).to(device)

    checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    prior_summary = checkpoint["summary"]
    config = prior_summary["config"]
    model = LatentImageAutoencoder(
        latent_channels=int(config["latent_channels"]),
        base_channels=int(config["base_channels"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False).to(device), dino, prior_summary


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if not 2 <= count <= length:
        raise ValueError("sheet step count must lie between two and the dense grid size")
    return torch.linspace(0, length - 1, count).round().long().tolist()


def save_diagnostics(
    probability: Tensor, speed: Tensor, output: Path
) -> None:
    time = torch.linspace(0.0, 1.0, probability.shape[1]).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for row in range(len(probability)):
        axes[0].plot(time, probability[row].numpy(), alpha=0.75)
        axes[1].plot(time[1:], speed[row].numpy(), alpha=0.75)
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[0].set(title="AFHQ DINO probe", xlabel="latent-chord time", ylabel="P(dog)", ylim=(-0.02, 1.02))
    axes[1].set(title="DINO feature speed", xlabel="latent-chord time", ylabel=r"$\|f_{k+1}-f_k\|_2$")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this demo requires Apple MPS")
    device = torch.device("mps")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, dino, prior_summary = load_models(args, device)
    probe, direction, probe_metrics = fit_or_load_afhq_probe(dino, args, device)

    cats = AFHQDomainDataset(
        args.afhq_root, "cat", image_size=128, training=False, augment=False
    )
    dogs = AFHQDomainDataset(
        args.afhq_root, "dog", image_size=128, training=False, augment=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    cat_indices = torch.randperm(len(cats), generator=generator)[: args.examples].tolist()
    dog_indices = torch.randperm(len(dogs), generator=generator)[: args.dog_pool].tolist()
    cat_tensor = torch.stack([cats[index] for index in cat_indices])
    dog_pool = torch.stack([dogs[index] for index in dog_indices])

    preprocess = BatchPreprocessor(64)
    cat_global, cat_patch, _ = dino_encode(dino, cat_tensor, preprocess, device)
    dog_global, dog_patch, _ = dino_encode(dino, dog_pool, preprocess, device)
    cat_descriptor = compact_spatial_content_descriptor(cat_global, cat_patch, direction)
    dog_descriptor = compact_spatial_content_descriptor(dog_global, dog_patch, direction)
    cost = content_cost_matrix(cat_descriptor, dog_descriptor)
    assignment = unique_minimum_assignment(cost)
    paired_dogs = dog_pool[assignment]

    endpoints = torch.cat([cat_tensor, paired_dogs]).to(device)
    endpoint_latent = model.encode(endpoints)
    cat_latent, dog_latent = endpoint_latent.chunk(2)
    time = torch.linspace(0.0, 1.0, args.display_steps, device=device)
    chord = torch.lerp(
        cat_latent[None], dog_latent[None], time[:, None, None, None, None]
    )
    decoded = model.decode(chord.flatten(0, 1)).reshape(
        args.display_steps, args.examples, 3, 128, 128
    ).transpose(0, 1)

    path_pil = [
        [
            tensor_to_pil(frame.mul(2).sub(1)).resize(
                (256, 256), Image.Resampling.LANCZOS
            )
            for frame in path
        ]
        for path in decoded.cpu()
    ]
    source_pil = [
        tensor_to_pil(image.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
        for image in cat_tensor
    ]
    reference_pil = [
        tensor_to_pil(image.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
        for image in paired_dogs
    ]
    sheet_index = evenly_spaced_indices(args.display_steps, args.sheet_steps)
    make_contact_sheet(
        source_pil,
        [[path[index] for index in sheet_index] for path in path_pil],
        reference_pil,
        args.output_dir / "heldout_latent_chord_contact_sheet.png",
        clock_label="latent chord",
    )
    make_distribution_gif(
        path_pil,
        args.output_dir / "heldout_latent_chord.gif",
        clock_label="latent chord",
        duration_ms=args.frame_duration_ms,
    )

    rendered_global, _, _ = dino_encode(
        dino, decoded.flatten(0, 1).cpu(), preprocess, device
    )
    rendered_global = rendered_global.reshape(args.examples, args.display_steps, -1)
    probability = probe.probabilities(rendered_global.flatten(0, 1)).reshape(
        args.examples, args.display_steps
    )
    feature_speed = torch.linalg.vector_norm(
        rendered_global[:, 1:] - rendered_global[:, :-1], dim=2
    )
    probability_delta = probability[:, 1:] - probability[:, :-1]
    total_variation = probability_delta.abs().sum(1)
    net_change = (probability[:, -1] - probability[:, 0]).abs()
    monotone_step_fraction = float((probability_delta >= -0.01).float().mean())
    monotone_path_fraction = float(
        (probability_delta >= -0.01).all(1).float().mean()
    )
    save_diagnostics(
        probability, feature_speed, args.output_dir / "heldout_path_diagnostics.png"
    )

    source_probability = probe.probabilities(cat_global)
    paired_dog_global = dog_global[assignment]
    reference_probability = probe.probabilities(paired_dog_global)
    adjacent_l1 = (decoded[:, 1:] - decoded[:, :-1]).abs().mean()
    summary: dict[str, object] = {
        "claim": "DINO-guided OT endpoint pairing with decoded latent-chord interpolation",
        "not_claimed": "No learned drift is integrated and the rendered path is not an SDE sample.",
        "device": str(device),
        "seed": args.seed,
        "display_steps": args.display_steps,
        "sheet_steps": args.sheet_steps,
        "cat_test_indices": cat_indices,
        "cat_test_names": [cats.paths[index].name for index in cat_indices],
        "dog_pool_test_indices": dog_indices,
        "paired_dog_pool_indices": assignment,
        "paired_dog_names": [dogs.paths[dog_indices[index]].name for index in assignment],
        "paired_content_cost": [float(cost[row, column]) for row, column in enumerate(assignment)],
        "afhq_probe_test_metrics": probe_metrics,
        "source_dog_probability": source_probability.tolist(),
        "reference_dog_probability": reference_probability.tolist(),
        "path_dog_probability": probability.tolist(),
        "semantic_monotone_step_fraction": monotone_step_fraction,
        "semantic_monotone_path_fraction": monotone_path_fraction,
        "semantic_total_variation_excess_mean": float((total_variation - net_change).mean()),
        "mean_dino_feature_step": float(feature_speed.mean()),
        "coefficient_of_variation_dino_speed": float(
            feature_speed.std(unbiased=False) / feature_speed.mean().clamp_min(1e-8)
        ),
        "mean_adjacent_frame_l1": float(adjacent_l1),
        "training_quarantine_verified": prior_summary["quarantine_verified"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {
            "trajectory_128": decoded.cpu(),
            "cat_test_indices": cat_indices,
            "dog_pool_test_indices": dog_indices,
            "assignment": assignment,
        },
        args.output_dir / "heldout_latent_chord_trajectories.pt",
    )
    results = f"""# Full-dataset AFHQ latent-chord demo

This is an uncurated, seeded held-out visualization of **DINO-guided OT
endpoint pairing followed by decoded latent-chord interpolation**. It is not
presented as an integrated learned SDE.

- AFHQ probe balanced test accuracy: {probe_metrics['balanced_accuracy']:.4f}
- semantic monotone step fraction: {monotone_step_fraction:.4f}
- semantic monotone path fraction: {monotone_path_fraction:.4f}
- mean adjacent-frame L1: {float(adjacent_l1):.5f}
- frames per animated path: {args.display_steps}
- seed: {args.seed}

The exact test indices and all per-frame probe probabilities are recorded in
`summary.json`.
"""
    (args.output_dir / "RESULTS.md").write_text(results)
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=project / "outputs/full_dataset_latent_bridge/full_dataset_latent_bridge.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "outputs/full_dataset_latent_bridge_demo",
    )
    parser.add_argument("--probe-cache", type=Path)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--frame-duration-ms", type=int, default=160)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--dog-pool", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.probe_cache is None:
        args.probe_cache = args.output_dir / "afhq_dino_linear_probe.pt"
    if args.examples > args.dog_pool:
        parser.error("--dog-pool must be at least --examples")
    if args.display_steps < 2 or not 2 <= args.sheet_steps <= args.display_steps:
        parser.error("invalid dense/sheet step counts")
    return args


if __name__ == "__main__":
    configuration = parse_args()
    configuration.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(render(configuration), indent=2))
