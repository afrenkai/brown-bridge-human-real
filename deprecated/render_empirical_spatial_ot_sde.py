"""Source-only empirical spatial-OT kernel with an exact pinned latent SDE."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.features import compact_spatial_content_descriptor
from brown_bridge.spatial_stochastic import sample_pinned_brownian_bridge
from render_dino_stargan_bridge import dino_encode
from render_full_dataset_demo import fit_or_load_afhq_probe
from train_bidirectional_kl_vae_diffusion import (
    decode_batches,
    decode_path,
    encode_batches,
    latent_cache,
    load_dino,
    load_vae,
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


def standardized_layout_features(
    value: Tensor,
    *,
    pool: int,
    mean: Tensor,
    scale: Tensor,
) -> Tensor:
    pooled = F.avg_pool2d(value.float(), pool).flatten(1)
    return F.normalize((pooled - mean) / scale, dim=1)


def unique_retrieval(cost: Tensor) -> Tensor:
    """Greedily round a rectangular nearest-neighbor cost to unique targets."""
    rows, columns = cost.shape
    used_row = torch.zeros(rows, dtype=torch.bool)
    used_column = torch.zeros(columns, dtype=torch.bool)
    assignment = torch.full((rows,), -1, dtype=torch.long)
    for edge in torch.argsort(cost.flatten()).tolist():
        row, column = divmod(edge, columns)
        if not used_row[row] and not used_column[column]:
            assignment[row] = column
            used_row[row] = True
            used_column[column] = True
            if bool(used_row.all()):
                break
    if torch.any(assignment < 0):
        raise RuntimeError("retrieval failed to assign every source")
    return assignment


@torch.inference_mode()
def heldout_spatial_descriptor(
    images: Tensor,
    dino: torch.nn.Module,
    preprocess: BatchPreprocessor,
    species: Tensor,
    device: torch.device,
) -> Tensor:
    global_features, patches, _ = dino_encode(dino, images, preprocess, device)
    return compact_spatial_content_descriptor(
        global_features, patches, species
    )


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
    latents, _, _ = latent_cache(vae, args, device)
    spatial_cache = torch.load(
        args.spatial_ot_feature_cache, map_location="cpu", weights_only=False
    )
    training_descriptor = {
        domain: F.normalize(spatial_cache[domain].float(), dim=1)
        for domain in ("cat", "dog")
    }
    spatial_species_direction = spatial_cache["species_direction"].float()

    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=128, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    generator = torch.Generator().manual_seed(args.seed + 80)
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
        )
        for domain, image in images.items()
    }
    heldout_descriptor = {
        domain: F.normalize(
            heldout_spatial_descriptor(
                image, dino, preprocess, spatial_species_direction, device
            ),
            dim=1,
        )
        for domain, image in images.items()
    }
    training_layout = {
        domain: F.avg_pool2d(latents[domain].float(), args.layout_pool).flatten(1)
        for domain in ("cat", "dog")
    }
    joined_layout = torch.cat([training_layout["cat"], training_layout["dog"]])
    layout_mean = joined_layout.mean(0)
    layout_scale = joined_layout.std(0, unbiased=False).clamp_min(1e-4)
    training_layout = {
        domain: F.normalize(
            (value - layout_mean) / layout_scale, dim=1
        )
        for domain, value in training_layout.items()
    }
    heldout_layout = {
        domain: standardized_layout_features(
            value,
            pool=args.layout_pool,
            mean=layout_mean,
            scale=layout_scale,
        )
        for domain, value in encoded.items()
    }

    retrieval: dict[str, Tensor] = {}
    for source_domain, target_domain, name in (
        ("cat", "dog", "cat_to_dog"),
        ("dog", "cat", "dog_to_cat"),
    ):
        semantic_cost = 1.0 - (
            heldout_descriptor[source_domain]
            @ training_descriptor[target_domain].T
        )
        layout_cost = 1.0 - (
            heldout_layout[source_domain] @ training_layout[target_domain].T
        )
        cost = (
            (1.0 - args.layout_weight) * semantic_cost
            + args.layout_weight * layout_cost
        )
        retrieval[name] = unique_retrieval(cost)

    target_latents = {
        "cat_to_dog": latents["dog"][retrieval["cat_to_dog"]].to(device),
        "dog_to_cat": latents["cat"][retrieval["dog_to_cat"]].to(device),
    }
    source_latents = {
        "cat_to_dog": encoded["cat"].to(device),
        "dog_to_cat": encoded["dog"].to(device),
    }
    paths = {
        name: sample_pinned_brownian_bridge(
            source.flatten(1),
            target_latents[name].flatten(1),
            epsilon=args.bridge_epsilon,
            macro_steps=args.display_steps - 1,
            microsteps_per_macro=args.microsteps,
        ).reshape(
            args.display_steps,
            len(source),
            *source.shape[1:],
        )
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
            vae,
            value,
            batch_size=args.vae_batch_size,
            device=device,
        )
        for name, value in target_latents.items()
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
            decoded_display[name],
            dino,
            probe,
            preprocess,
            device,
            direction=direction,
        )
        probabilities[name] = probability
        endpoint_features = dino_encode(
            dino, decoded_endpoints[name], preprocess, device
        )[0]
        endpoint_probability = probe.probabilities(endpoint_features)
        result.update(
            {
                "all_heldout_source_dog_probability_mean": float(
                    probe.probabilities(real_features[source_domain]).mean()
                ),
                "all_endpoint_dog_probability_mean": float(
                    endpoint_probability.mean()
                ),
                "all_endpoint_classified_as_target_fraction": float(
                    (
                        (endpoint_probability >= 0.5)
                        if direction > 0
                        else (endpoint_probability < 0.5)
                    ).float().mean()
                ),
                "retrieved_semantic_cosine_mean": float(
                    (
                        heldout_descriptor[source_domain]
                        * training_descriptor[target_domain][retrieval[name]]
                    ).sum(1).mean()
                ),
            }
        )
        result.update(
            endpoint_distribution_metrics(
                real_features[source_domain],
                endpoint_features,
                real_features[target_domain],
                species_probe_direction,
                target_name=f"{target_domain}s",
            )
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
        make_domain_contact_sheet(
            pil_images(images[source_domain][: args.examples]),
            [[row[index] for index in sheet_indices] for row in path_images],
            pil_images(decoded_endpoints[name][: args.examples]),
            args.output_dir / f"empirical_ot_sde_{name}.png",
            source_name=source_domain,
            target_name=target_domain,
            clock_label="exact pinned Brownian SDE after spatial-DINO support retrieval",
            reference_label=f"retrieved training {target_domain} endpoint",
        )
        make_path_gif(
            path_images,
            args.output_dir / f"empirical_ot_sde_{name}.gif",
            source_name=source_domain,
            target_name=target_domain,
        )

    summary = {
        "method": "empirical spatial-DINO OT kernel plus exact pinned Brownian SDE",
        "source_input_only": True,
        "uses_retrieved_training_support_endpoint": True,
        "user_or_heldout_target_supplied": False,
        "nonparametric_coherence_ceiling_not_generalization_claim": True,
        "training_support_counts": {
            "cats": len(latents["cat"]),
            "dogs": len(latents["dog"]),
        },
        "heldout_counts": {domain: len(value) for domain, value in images.items()},
        "metrics": metrics,
        "probe_test": probe_test,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(retrieval, args.output_dir / "retrieved_training_indices.pt")
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
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/empirical_spatial_ot_sde")
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bridge-epsilon", type=float, default=0.01)
    parser.add_argument("--layout-weight", type=float, default=0.2)
    parser.add_argument("--layout-pool", type=int, default=2)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--display-steps", type=int, default=21)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--sheet-steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
