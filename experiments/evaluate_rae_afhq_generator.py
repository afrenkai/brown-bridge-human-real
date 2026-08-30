from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, ToTensor
from utils.model_utils import instantiate_from_config
from utils.train_utils import center_crop_arr, parse_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def extract_features(
    model: nn.Module,
    root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, object]:
    transform = Compose(
        (
            Lambda(lambda image: center_crop_arr(image, 256)),
            ToTensor(),
        )
    )
    dataset = ImageFolder(root, transform=transform)
    if dataset.classes != ["cat", "dog"]:
        raise ValueError(f"Expected cat/dog classes at {root}, found {dataset.classes}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latent = model.encode(images)
            pooled = latent.float().mean(dim=(2, 3)).cpu()
            feature_batches.append(pooled)
            label_batches.append(labels)
    return {
        "features": torch.cat(feature_batches),
        "labels": torch.cat(label_batches),
        "paths": [sample_path for sample_path, sample_label in dataset.samples],
        "classes": dataset.classes,
        "root": str(root),
    }


def feature_class(split: dict[str, object], class_label: int) -> torch.Tensor:
    features = split["features"]
    labels = split["labels"]
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Feature split tensors are invalid")
    return features[labels == class_label]


def fit_projection(
    train_features: torch.Tensor,
    dimension: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    mean = train_features.mean(dim=0)
    centered = (train_features - mean).to("cuda")
    maximum_dimension = min(centered.shape) - 1
    projection_dimension = min(dimension, maximum_dimension)
    left_vectors, singular_values, components = torch.pca_lowrank(
        centered,
        q=projection_dimension,
        center=False,
        niter=5,
    )
    del left_vectors, singular_values
    return mean, components.cpu()


def project_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    return (features - mean) @ components


def aggregate_metric_records(records: list[dict[str, float]]) -> dict[str, object]:
    if not records:
        raise ValueError("At least one metric record is required")
    aggregate: dict[str, object] = {"matched_shards": records}
    for key in records[0]:
        values = torch.tensor([record[key] for record in records])
        aggregate[key] = float(values.mean())
        aggregate[f"{key}_p05"] = float(torch.quantile(values, 0.05))
        aggregate[f"{key}_p95"] = float(torch.quantile(values, 0.95))
    return aggregate


def matched_distribution_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
) -> dict[str, object]:
    shard_size = len(real)
    shard_count = len(generated) // shard_size
    if shard_count < 1:
        raise ValueError("Generated split must be at least as large as the real split")
    records = [
        distribution_metrics(
            real,
            generated[shard_index * shard_size : (shard_index + 1) * shard_size],
        )
        for shard_index in range(shard_count)
    ]
    result = aggregate_metric_records(records)
    result["real_count_per_shard"] = shard_size
    result["generated_count_per_shard"] = shard_size
    result["shard_count"] = shard_count
    return result


def matched_diversity_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
) -> dict[str, object]:
    shard_size = len(real)
    shard_count = len(generated) // shard_size
    real_distance = off_diagonal_values(pairwise_cosine_distance(real, real))
    real_summary = summary(real_distance)
    generated_summaries: list[dict[str, float]] = []
    ratios: list[float] = []
    for shard_index in range(shard_count):
        shard = generated[shard_index * shard_size : (shard_index + 1) * shard_size]
        shard_distance = off_diagonal_values(pairwise_cosine_distance(shard, shard))
        generated_summary = summary(shard_distance)
        generated_summaries.append(generated_summary)
        ratios.append(generated_summary["mean"] / real_summary["mean"])
    ratio_values = torch.tensor(ratios)
    return {
        "heldout_real": real_summary,
        "generated_matched_shards": generated_summaries,
        "mean_ratio_generated_to_real": float(ratio_values.mean()),
        "mean_ratio_p05": float(torch.quantile(ratio_values, 0.05)),
        "mean_ratio_p95": float(torch.quantile(ratio_values, 0.95)),
        "shard_count": shard_count,
        "count_per_shard": shard_size,
    }


def fast_eigendecomposition(
    matrix: torch.Tensor,
    eigenvectors_required: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    symmetric = (matrix + matrix.T) * 0.5
    finite = bool(torch.isfinite(symmetric).all())
    maximum_magnitude = float(symmetric.abs().max()) if symmetric.numel() else 0.0
    solve = (
        symmetric.float().to("cuda")
        if finite and maximum_magnitude <= 1.0e12
        else symmetric.double().cpu()
    )
    eigenvalues: torch.Tensor | None = None
    eigenvectors: torch.Tensor | None = None
    try:
        if eigenvectors_required:
            eigenvalues, eigenvectors = torch.linalg.eigh(solve)
        else:
            eigenvalues = torch.linalg.eigvalsh(solve)
    except torch.linalg.LinAlgError:
        pass
    valid_values = eigenvalues is not None and bool(torch.isfinite(eigenvalues).all())
    valid_vectors = (
        not eigenvectors_required
        or eigenvectors is not None
        and bool(torch.isfinite(eigenvectors).all())
    )
    if not valid_values or not valid_vectors:
        retry = symmetric.double().cpu()
        scale = retry.diagonal().abs().mean().clamp_min(1.0)
        ridge = torch.finfo(retry.dtype).eps * scale
        retry = retry + ridge * torch.eye(retry.shape[0], dtype=retry.dtype)
        if eigenvectors_required:
            eigenvalues, eigenvectors = torch.linalg.eigh(retry)
        else:
            eigenvalues = torch.linalg.eigvalsh(retry)
    if eigenvalues is None:
        raise RuntimeError("Symmetric eigendecomposition did not return eigenvalues")
    return eigenvalues.clamp_min(0), eigenvectors


def frechet_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    first_values = first.float().to("cuda")
    second_values = second.float().to("cuda")
    first_values = first_values[torch.isfinite(first_values).all(dim=1)]
    second_values = second_values[torch.isfinite(second_values).all(dim=1)]
    if len(first_values) < 2 or len(second_values) < 2:
        return float("nan")
    first_mean = first_values.mean(dim=0)
    second_mean = second_values.mean(dim=0)
    first_centered = first_values - first_mean
    second_centered = second_values - second_mean
    first_covariance = first_centered.T @ first_centered / (len(first_values) - 1)
    second_covariance = second_centered.T @ second_centered / (len(second_values) - 1)
    eigenvalues, eigenvectors = fast_eigendecomposition(
        first_covariance,
        eigenvectors_required=True,
    )
    if eigenvectors is None:
        raise RuntimeError("Fréchet eigendecomposition requires eigenvectors")
    working_device = eigenvectors.device
    working_dtype = eigenvectors.dtype
    first_covariance = first_covariance.to(working_device, working_dtype)
    second_covariance = second_covariance.to(working_device, working_dtype)
    first_root = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    inner = first_root @ second_covariance @ first_root
    inner_eigenvalues, inner_eigenvectors = fast_eigendecomposition(
        inner,
        eigenvectors_required=False,
    )
    del inner_eigenvectors
    covariance_root_trace = (
        inner_eigenvalues.sqrt()
        .sum()
        .to(
            working_device,
            working_dtype,
        )
    )
    mean_distance = (
        (
            first_mean.to(working_device, working_dtype)
            - second_mean.to(working_device, working_dtype)
        )
        .square()
        .sum()
    )
    distance = (
        mean_distance
        + first_covariance.trace()
        + second_covariance.trace()
        - 2.0 * covariance_root_trace
    )
    return float(distance.clamp_min(0).item())


def pairwise_cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_normalized = functional.normalize(first.float(), dim=1)
    second_normalized = functional.normalize(second.float(), dim=1)
    return (1.0 - first_normalized @ second_normalized.T).clamp_min(0)


def sinkhorn_cost(cost: torch.Tensor, epsilon: float, iterations: int) -> float:
    row_count, column_count = cost.shape
    row_log_mass = -math.log(row_count)
    column_log_mass = -math.log(column_count)
    row_potential = torch.zeros(row_count, device=cost.device)
    column_potential = torch.zeros(column_count, device=cost.device)
    iteration_count = 0
    while iteration_count < iterations:
        row_potential = epsilon * (
            row_log_mass
            - torch.logsumexp(
                (column_potential.unsqueeze(0) - cost) / epsilon,
                dim=1,
            )
        )
        column_potential = epsilon * (
            column_log_mass
            - torch.logsumexp(
                (row_potential.unsqueeze(1) - cost) / epsilon,
                dim=0,
            )
        )
        iteration_count += 1
    transport = torch.exp(
        (row_potential.unsqueeze(1) + column_potential.unsqueeze(0) - cost) / epsilon
    )
    return float((transport * cost).sum().item())


def distribution_metrics(
    real: torch.Tensor, generated: torch.Tensor
) -> dict[str, float]:
    device = torch.device("cuda")
    real_device = real.to(device)
    generated_device = generated.to(device)
    real_generated_cost = pairwise_cosine_distance(real_device, generated_device)
    real_real_cost = pairwise_cosine_distance(real_device, real_device)
    generated_generated_cost = pairwise_cosine_distance(
        generated_device, generated_device
    )
    epsilon = 0.05
    iterations = 100
    cross_sinkhorn = sinkhorn_cost(real_generated_cost, epsilon, iterations)
    real_sinkhorn = sinkhorn_cost(real_real_cost, epsilon, iterations)
    generated_sinkhorn = sinkhorn_cost(generated_generated_cost, epsilon, iterations)
    sinkhorn_divergence = cross_sinkhorn - 0.5 * (real_sinkhorn + generated_sinkhorn)
    real_real_distance = real_real_cost.sqrt()
    generated_generated_distance = generated_generated_cost.sqrt()
    real_count = len(real_real_distance)
    generated_count = len(generated_generated_distance)
    real_within_mean = (
        real_real_distance.sum() - real_real_distance.diagonal().sum()
    ) / (real_count * (real_count - 1))
    generated_within_mean = (
        generated_generated_distance.sum()
        - generated_generated_distance.diagonal().sum()
    ) / (generated_count * (generated_count - 1))
    energy_distance = float(
        2.0 * real_generated_cost.sqrt().mean()
        - real_within_mean
        - generated_within_mean
    )
    return {
        "rae_fd128": frechet_distance(real, generated),
        "sinkhorn_divergence_cosine": max(0.0, sinkhorn_divergence),
        "energy_distance_cosine_unbiased": energy_distance,
    }


def off_diagonal_values(distance: torch.Tensor) -> torch.Tensor:
    row_count = distance.shape[0]
    mask = ~torch.eye(row_count, dtype=torch.bool, device=distance.device)
    return distance[mask]


def summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p01": float(torch.quantile(values, 0.01)),
        "p05": float(torch.quantile(values, 0.05)),
        "p95": float(torch.quantile(values, 0.95)),
    }


def nearest_distances(
    query: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = pairwise_cosine_distance(query, reference)
    return distance.min(dim=1)


def memorization_metrics(
    train: torch.Tensor,
    validation: torch.Tensor,
    generated: torch.Tensor,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    generated_distances, generated_indices = nearest_distances(generated, train)
    validation_distances, validation_indices = nearest_distances(validation, train)
    del validation_indices
    generated_summary = summary(generated_distances)
    validation_summary = summary(validation_distances)
    return (
        {
            "generated_to_train": generated_summary,
            "heldout_to_train": validation_summary,
            "mean_ratio_generated_to_heldout": generated_summary["mean"]
            / validation_summary["mean"],
            "near_exact_generated_count": int((generated_distances < 1.0e-4).sum()),
        },
        generated_distances,
        generated_indices,
    )


def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    steps: int,
    seed: int,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, list[float]]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1.0e-5)
    normalized_features = ((train_features - feature_mean) / feature_std).to(device)
    labels = train_labels.to(device)
    model = nn.Linear(train_features.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1.0e-4)
    losses: list[float] = []
    for step in range(steps):
        logits = model(normalized_features)
        loss = functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step + 1 == steps:
            losses.append(float(loss.detach()))
    return model.eval(), feature_mean, feature_std, losses


def probe_metrics(
    model: nn.Linear,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    split: dict[str, object],
) -> dict[str, object]:
    features = split["features"]
    labels = split["labels"]
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Feature split tensors are invalid")
    device = torch.device("cuda")
    with torch.inference_mode():
        logits = model(((features - feature_mean) / feature_std).to(device))
        predictions = logits.argmax(dim=1).cpu()
    class_accuracy: dict[str, float] = {}
    for class_label, class_name in ((0, "cat"), (1, "dog")):
        class_mask = labels == class_label
        class_accuracy[class_name] = float(
            (predictions[class_mask] == labels[class_mask]).float().mean()
        )
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "per_class_accuracy": class_accuracy,
        "confusion_counts": [
            [
                int(((labels == true_label) & (predictions == predicted_label)).sum())
                for predicted_label in (0, 1)
            ]
            for true_label in (0, 1)
        ],
    }


def split_paths_for_class(split: dict[str, object], class_label: int) -> list[Path]:
    labels = split["labels"]
    paths = split["paths"]
    if not isinstance(labels, torch.Tensor) or not isinstance(paths, list):
        raise TypeError("Feature split paths are invalid")
    label_values = labels.tolist()
    if len(paths) != len(label_values):
        raise ValueError("Feature split path and label counts differ")
    return [
        Path(path)
        for path_index, path in enumerate(paths)
        if label_values[path_index] == class_label
    ]


def save_nearest_montage(
    generated_paths: list[Path],
    train_paths: list[Path],
    distances: torch.Tensor,
    indices: torch.Tensor,
    output_path: Path,
) -> None:
    selected = torch.argsort(distances)[:8]
    image_size = 256
    montage = Image.new("RGB", (image_size * 2, image_size * len(selected)))
    resampling = Image.Resampling.LANCZOS
    for row, selected_index in enumerate(selected.tolist()):
        generated = Image.open(generated_paths[selected_index]).convert("RGB")
        neighbor = Image.open(train_paths[int(indices[selected_index])]).convert("RGB")
        generated.thumbnail((image_size, image_size), resampling)
        neighbor.thumbnail((image_size, image_size), resampling)
        montage.paste(generated, (0, row * image_size))
        montage.paste(neighbor, (image_size, row * image_size))
    montage.save(output_path)


def build_report(metrics: dict[str, object]) -> str:
    lines = [
        "# RAE AFHQ held-out evaluation",
        "",
        "All real evaluation images are from the untouched AFHQ validation split.",
        "RAE-FD128 uses train-fitted PCA-128 of spatially mean-pooled frozen normalized RAE encoder features.",
        "",
        f"- Validation probe accuracy: {metrics['probe']['validation']['accuracy']:.4f}",
        f"- Generated class accuracy: {metrics['probe']['generated']['accuracy']:.4f}",
        "",
        "| Class | RAE-FD128 | Sinkhorn | Diversity ratio | NN ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    class_metrics = metrics["classes"]
    if not isinstance(class_metrics, dict):
        raise TypeError("Class metrics are invalid")
    for class_name in ("cat", "dog"):
        values = class_metrics[class_name]
        lines.append(
            f"| {class_name} | {values['distribution']['rae_fd128']:.4f} | "
            f"{values['distribution']['sinkhorn_divergence_cosine']:.4f} | "
            f"{values['diversity']['mean_ratio_generated_to_real']:.4f} | "
            f"{values['memorization']['mean_ratio_generated_to_heldout']:.4f} |"
        )
    lines.extend(
        (
            "",
            (
                "Nearest-neighbor ratios compare generated-to-train distance with "
                "heldout-to-train distance. Values near one do not indicate extra memorization."
            ),
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    config = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        evaluation_config,
    ) = parse_configs(config)
    del (
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        evaluation_config,
    )
    rae = instantiate_from_config(rae_config).to(device).eval()
    started_at = perf_counter()
    splits: dict[str, dict[str, object]] = {}
    for split_name, root in (
        ("train", args.train_root),
        ("validation", args.validation_root),
        ("generated", args.generated_root),
    ):
        split = extract_features(
            rae,
            root,
            args.batch_size,
            args.num_workers,
            device,
        )
        splits[split_name] = split
        torch.save(split, feature_dir / f"{split_name}.pt")
        print(f"Encoded {len(split['labels'])} {split_name} images")
    train_features = splits["train"]["features"]
    train_labels = splits["train"]["labels"]
    if not isinstance(train_features, torch.Tensor) or not isinstance(
        train_labels, torch.Tensor
    ):
        raise TypeError("Training feature tensors are invalid")
    projection_mean, projection_components = fit_projection(
        train_features,
        128,
        args.seed,
    )
    torch.save(
        {"mean": projection_mean, "components": projection_components},
        args.output_dir / "train_pca128.pt",
    )
    projected_splits: dict[str, dict[str, object]] = {}
    for split_name, split in splits.items():
        features = split["features"]
        if not isinstance(features, torch.Tensor):
            raise TypeError("Feature split tensor is invalid")
        projected_split = dict(split)
        projected_split["features"] = project_features(
            features,
            projection_mean,
            projection_components,
        )
        projected_splits[split_name] = projected_split
    torch.set_grad_enabled(True)
    probe, feature_mean, feature_std, probe_losses = train_probe(
        train_features,
        train_labels,
        args.probe_steps,
        args.seed,
    )
    torch.set_grad_enabled(False)
    metrics: dict[str, object] = {
        "known_classes": ["cat", "dog"],
        "heldout_split": str(args.validation_root),
        "counts": {
            split_name: len(split["labels"]) for split_name, split in splits.items()
        },
        "probe": {
            "optimization_losses": probe_losses,
            "train": probe_metrics(probe, feature_mean, feature_std, splits["train"]),
            "validation": probe_metrics(
                probe, feature_mean, feature_std, splits["validation"]
            ),
            "generated": probe_metrics(
                probe, feature_mean, feature_std, splits["generated"]
            ),
        },
        "classes": {},
    }
    class_results = metrics["classes"]
    if not isinstance(class_results, dict):
        raise TypeError("Class result container is invalid")
    for class_label, class_name in ((0, "cat"), (1, "dog")):
        train_class = feature_class(splits["train"], class_label)
        validation_class = feature_class(splits["validation"], class_label)
        generated_class = feature_class(splits["generated"], class_label)
        projected_validation_class = feature_class(
            projected_splits["validation"], class_label
        )
        projected_generated_class = feature_class(
            projected_splits["generated"], class_label
        )
        distribution = matched_distribution_metrics(
            projected_validation_class,
            projected_generated_class,
        )
        diversity = matched_diversity_metrics(
            projected_validation_class,
            projected_generated_class,
        )
        memorization, nearest_distance, nearest_index = memorization_metrics(
            train_class,
            validation_class,
            generated_class,
        )
        class_results[class_name] = {
            "counts": {
                "train": len(train_class),
                "validation": len(validation_class),
                "generated": len(generated_class),
            },
            "distribution": distribution,
            "diversity": diversity,
            "memorization": memorization,
        }
        save_nearest_montage(
            split_paths_for_class(splits["generated"], class_label),
            split_paths_for_class(splits["train"], class_label),
            nearest_distance,
            nearest_index,
            args.output_dir / f"nearest_train_{class_name}.png",
        )
    metrics["elapsed_seconds"] = perf_counter() - started_at
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "report.md").write_text(build_report(metrics))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
