"""Pure entropic-OT displacement interpolation between MNIST classes on MPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets

from brown_bridge.ot import sinkhorn
from brown_bridge.paths import sample_coupling
from brown_bridge.schemas import CoupledEndpoints


def load_mnist_class(
    root: Path,
    *,
    label: int,
    sample_count: int,
    image_size: int,
    download: bool,
) -> tuple[Tensor, tuple[int, int, int]]:
    """Load one MNIST class as flattened images and preserve its image shape."""
    if not 0 <= label <= 9:
        raise ValueError("MNIST labels must lie in [0, 9]")
    if sample_count < 1 or image_size < 4:
        raise ValueError(
            "sample_count must be positive and image_size must be at least four"
        )
    dataset = datasets.MNIST(root=root, train=True, download=download)
    selected = dataset.targets == label
    images = dataset.data[selected][:sample_count].float().div(255.0).unsqueeze(1)
    if len(images) < sample_count:
        raise ValueError(f"MNIST class {label} has only {len(images)} samples")
    if image_size != images.shape[-1]:
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return images.flatten(1), tuple(images.shape[1:])


def displacement_path(
    endpoints: CoupledEndpoints,
    times: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    """Evaluate exact linear displacement interpolation on the MPS device."""
    source = endpoints.source.to(device)
    target = endpoints.target.to(device)
    time = times[:, None, None]
    return (1.0 - time) * source[None] + time * target[None]


def pca_coordinates(
    endpoint_source: Tensor,
    endpoint_target: Tensor,
    midpoint: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project source, midpoint, and target empirical clouds to two PCs."""
    all_features = torch.cat([endpoint_source, midpoint, endpoint_target])
    centered = all_features - all_features.mean(0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=2, center=False)
    coordinates = centered @ components[:, :2]
    count = len(endpoint_source)
    return coordinates[:count], coordinates[count : 2 * count], coordinates[2 * count :]


def plot_results(
    forward_path: Tensor,
    reverse_path: Tensor,
    forward_endpoints: CoupledEndpoints,
    output_path: Path,
    *,
    source_label: int,
    target_label: int,
) -> None:
    """Save representative exact chords and PCA cloud movement."""
    forward_cpu = forward_path.detach().cpu()
    reverse_cpu = reverse_path.detach().cpu()
    forward_source = forward_cpu[0]
    forward_midpoint = forward_cpu[len(forward_cpu) // 2]
    forward_target = forward_cpu[-1]
    reverse_source = reverse_cpu[0]
    reverse_midpoint = reverse_cpu[len(reverse_cpu) // 2]
    reverse_target = reverse_cpu[-1]
    source_coordinates, midpoint_coordinates, target_coordinates = pca_coordinates(
        forward_source,
        forward_target,
        forward_midpoint,
    )

    figure, axes = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)
    image_rows = (
        (
            (forward_source, forward_midpoint, forward_target),
            f"exact OT: {source_label} → {target_label}",
        ),
        (
            (reverse_source, reverse_midpoint, reverse_target),
            f"exact OT: {target_label} → {source_label}",
        ),
    )
    image_titles = ("source", "t = 0.5", "target")
    for column, title in enumerate(image_titles):
        axes[0, column].set_title(title)
    for row, (images, row_title) in enumerate(image_rows):
        axes[row, 0].set_ylabel(row_title)
        for column, image_batch in enumerate(images):
            axes[row, column].imshow(
                image_batch[0].reshape(28, 28),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, column].axis("off")

    axes[0, 3].scatter(
        source_coordinates[:, 0], source_coordinates[:, 1], s=8, label="source"
    )
    axes[0, 3].scatter(
        midpoint_coordinates[:, 0],
        midpoint_coordinates[:, 1],
        s=8,
        label="t = 0.5",
    )
    axes[0, 3].scatter(
        target_coordinates[:, 0], target_coordinates[:, 1], s=8, label="target"
    )
    axes[0, 3].set_title("PCA: forward cloud")
    axes[1, 3].scatter(
        target_coordinates[:, 0], target_coordinates[:, 1], s=8, label="source"
    )
    axes[1, 3].scatter(
        midpoint_coordinates[:, 0],
        midpoint_coordinates[:, 1],
        s=8,
        label="t = 0.5",
    )
    axes[1, 3].scatter(
        source_coordinates[:, 0], target_coordinates[:, 1], s=8, label="target"
    )
    axes[1, 3].set_title("PCA: reverse cloud")
    for axis in (axes[0, 3], axes[1, 3]):
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/mnist"))
    parser.add_argument("--source-label", type=int, default=9)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--particles", type=int, default=2_048)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--display-steps", type=int, default=9)
    parser.add_argument("--ot-epsilon", type=float, default=2.0)
    parser.add_argument("--ot-iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/mnist_mps_ot/mnist_mps_ot.png"),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this example requires an available Metal MPS device")
    if args.source_label == args.target_label:
        raise ValueError("source-label and target-label must differ")
    if args.display_steps < 2 or args.particles < 1:
        raise ValueError(
            "display-steps must be at least two and particles must be positive"
        )

    device = torch.device("mps")
    source, source_shape = load_mnist_class(
        args.data_root,
        label=args.source_label,
        sample_count=args.sample_count,
        image_size=args.image_size,
        download=args.download,
    )
    target, target_shape = load_mnist_class(
        args.data_root,
        label=args.target_label,
        sample_count=args.sample_count,
        image_size=args.image_size,
        download=args.download,
    )
    if source_shape != target_shape or source_shape != (1, 28, 28):
        raise ValueError("this visualization expects one-channel 28x28 images")
    ot_result = sinkhorn(
        source,
        target,
        epsilon=args.ot_epsilon,
        max_iterations=args.ot_iterations,
        cost="sqeuclidean",
    )
    endpoints = sample_coupling(
        ot_result.plan,
        source,
        target,
        args.particles,
        generator=torch.Generator().manual_seed(args.seed),
    )
    times = torch.linspace(0.0, 1.0, args.display_steps, device=device)
    forward_path = displacement_path(endpoints, times, device=device)
    reverse_endpoints = CoupledEndpoints(
        endpoints.target,
        endpoints.source,
        endpoints.target_indices,
        endpoints.source_indices,
    )
    reverse_path = displacement_path(reverse_endpoints, times, device=device)
    forward_terminal = forward_path[-1].cpu()
    reverse_terminal = reverse_path[-1].cpu()
    forward_check = sinkhorn(
        forward_terminal,
        target,
        epsilon=args.ot_epsilon,
        max_iterations=args.ot_iterations,
        cost="sqeuclidean",
    )
    reverse_check = sinkhorn(
        reverse_terminal,
        source,
        epsilon=args.ot_epsilon,
        max_iterations=args.ot_iterations,
        cost="sqeuclidean",
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(
        forward_path,
        reverse_path,
        endpoints,
        args.output_path,
        source_label=args.source_label,
        target_label=args.target_label,
    )
    summary = {
        "device": str(device),
        "method": "entropic OT coupling plus exact displacement interpolation",
        "feature_dimension": int(source.shape[1]),
        "source_label": args.source_label,
        "target_label": args.target_label,
        "sample_count": args.sample_count,
        "particles": args.particles,
        "ot_transport_cost": float(ot_result.transport_cost),
        "forward_terminal_transport_cost": float(forward_check.transport_cost),
        "reverse_terminal_transport_cost": float(reverse_check.transport_cost),
        "output_path": str(args.output_path),
    }
    args.output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
