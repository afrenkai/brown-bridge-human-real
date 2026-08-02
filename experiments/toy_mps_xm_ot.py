"""Small MPS sanity check for OT-paired bidirectional XM transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from brown_bridge.exploration import best_of_k_loss
from brown_bridge.ot import SinkhornResult, sinkhorn
from brown_bridge.paths import CoupledEndpoints, sample_coupling


class ToyTransportModel(nn.Module):
    """A small shared cat↔dog endpoint model for two-dimensional samples."""

    def __init__(self, *, hidden_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 2),
        )

    def forward(self, source: Tensor, noise: Tensor, direction: Tensor) -> Tensor:
        if source.ndim != 2 or noise.shape != source.shape:
            raise ValueError("source and noise must share shape [batch, 2]")
        if direction.numel() != len(source):
            raise ValueError("direction must have one value per sample")
        direction_column = direction.reshape(-1, 1).to(source)
        return self.network(torch.cat([source, noise, direction_column], dim=1))


def make_toy_samples(
    centers: Tensor,
    sample_count: int,
    *,
    noise_scale: float,
    generator: torch.Generator,
) -> Tensor:
    """Draw a balanced mixture of small two-dimensional Gaussian clusters."""
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape [components, 2]")
    if sample_count < 1 or noise_scale <= 0:
        raise ValueError("sample_count must be positive and noise_scale must be positive")
    component_indices = torch.randint(
        len(centers),
        (sample_count,),
        generator=generator,
    )
    component_noise = torch.randn(sample_count, 2, generator=generator)
    return centers[component_indices] + noise_scale * component_noise


def train_bidirectional_model(
    paired_endpoints: CoupledEndpoints,
    plan: Tensor,
    source: Tensor,
    target: Tensor,
    *,
    batch_size: int,
    candidate_count: int,
    hidden_width: int,
    learning_rate: float,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[ToyTransportModel, list[float]]:
    """Train one direction-conditioned model on sampled OT endpoint pairs."""
    if batch_size < 1 or candidate_count < 1 or steps < 1:
        raise ValueError("batch_size, candidate_count, and steps must be positive")
    if tuple(plan.shape) != (len(source), len(target)):
        raise ValueError("plan shape must match source and target sample counts")

    model = ToyTransportModel(hidden_width=hidden_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    model.train()

    for step in range(steps):
        pair_indices = torch.randint(
            len(paired_endpoints.source),
            (batch_size,),
            generator=generator,
        )
        paired_source = paired_endpoints.source[pair_indices]
        paired_target = paired_endpoints.target[pair_indices]
        direction = torch.where(
            torch.randint(0, 2, (batch_size,), generator=generator) == 0,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        forward = direction > 0
        source_batch = torch.where(
            forward[:, None], paired_source, paired_target
        ).to(device)
        target_batch = torch.where(
            forward[:, None], paired_target, paired_source
        ).to(device)
        direction = direction.to(device)

        noise = torch.randn(
            candidate_count,
            batch_size,
            2,
            generator=generator,
        ).to(device)
        candidate_source = source_batch[None, :, :].expand(candidate_count, -1, -1)
        candidate_direction = direction[None, :].expand(candidate_count, -1)
        candidate_predictions = model(
            candidate_source.reshape(-1, 2),
            noise.reshape(-1, 2),
            candidate_direction.reshape(-1),
        ).reshape(candidate_count, batch_size, 2)
        candidate_losses = (
            candidate_predictions - target_batch[None, :, :]
        ).square().mean(dim=2)
        selected = best_of_k_loss(candidate_losses.transpose(0, 1))

        optimizer.zero_grad(set_to_none=True)
        selected.loss.backward()
        optimizer.step()
        history.append(float(selected.loss.detach()))

        if step == 0 or (step + 1) % max(steps // 10, 1) == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": history[-1],
                        "candidate_count": candidate_count,
                    }
                ),
                flush=True,
            )
    return model.eval(), history


@torch.inference_mode()
def sample_direction(
    model: ToyTransportModel,
    source: Tensor,
    *,
    direction: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Sample one endpoint per source point in one transport direction."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(len(source), 2, generator=generator)
    direction_values = torch.full((len(source),), float(direction))
    return model(
        source.to(device),
        noise.to(device),
        direction_values.to(device),
    ).cpu()


def plot_results(
    source: Tensor,
    target: Tensor,
    paired_endpoints: CoupledEndpoints,
    forward_samples: Tensor,
    backward_samples: Tensor,
    history: list[float],
    output_path: Path,
) -> None:
    """Save endpoint, forward, reverse, and loss views in one figure."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    source_points = source.numpy()
    target_points = target.numpy()

    axes[0, 0].scatter(source_points[:, 0], source_points[:, 1], s=12, label="cat")
    axes[0, 0].scatter(target_points[:, 0], target_points[:, 1], s=12, label="dog")
    for pair_index in range(min(24, len(paired_endpoints.source))):
        pair_source = paired_endpoints.source[pair_index]
        pair_target = paired_endpoints.target[pair_index]
        axes[0, 0].plot(
            [pair_source[0], pair_target[0]],
            [pair_source[1], pair_target[1]],
            color="black",
            alpha=0.18,
            linewidth=0.7,
        )
    axes[0, 0].set_title("OT endpoint samples")
    axes[0, 0].legend()

    axes[0, 1].scatter(source_points[:, 0], source_points[:, 1], s=12, label="cat source")
    forward_points = forward_samples.numpy()
    axes[0, 1].scatter(
        forward_points[:, 0],
        forward_points[:, 1],
        s=12,
        label="XM cat→dog",
    )
    axes[0, 1].scatter(
        target_points[:, 0],
        target_points[:, 1],
        s=12,
        alpha=0.25,
        label="dog target",
    )
    axes[0, 1].set_title("Forward transport")
    axes[0, 1].legend()

    axes[1, 0].scatter(target_points[:, 0], target_points[:, 1], s=12, label="dog source")
    backward_points = backward_samples.numpy()
    axes[1, 0].scatter(
        backward_points[:, 0],
        backward_points[:, 1],
        s=12,
        label="XM dog→cat",
    )
    axes[1, 0].scatter(
        source_points[:, 0],
        source_points[:, 1],
        s=12,
        alpha=0.25,
        label="cat target",
    )
    axes[1, 0].set_title("Reverse transport")
    axes[1, 0].legend()

    axes[1, 1].plot(history)
    axes[1, 1].set_title("Best-of-K training loss")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("selected MSE")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    axes[1, 1].grid(alpha=0.2)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def summarize_transport(
    samples: Tensor,
    target: Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> SinkhornResult:
    """Measure generated samples against a target toy measure."""
    return sinkhorn(
        samples,
        target,
        epsilon=epsilon,
        max_iterations=iterations,
        cost="sqeuclidean",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--paired-samples", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--cluster-noise", type=float, default=0.22)
    parser.add_argument("--ot-epsilon", type=float, default=0.15)
    parser.add_argument("--ot-iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/toy_mps_xm_ot/toy_mps_xm_ot.png"),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal is unavailable: this example requires torch.device('mps')")
    if args.sample_count < 1 or args.paired_samples < 1:
        raise ValueError("sample-count and paired-samples must be positive")

    device = torch.device("mps")
    torch.manual_seed(args.seed)
    source_centers = torch.tensor([[-2.0, -0.6], [-1.2, 1.2], [0.0, -1.4]])
    target_centers = torch.tensor([[1.7, -1.0], [2.0, 1.0], [0.3, 1.6]])
    source_generator = torch.Generator().manual_seed(args.seed)
    target_generator = torch.Generator().manual_seed(args.seed + 1)
    source = make_toy_samples(
        source_centers,
        args.sample_count,
        noise_scale=args.cluster_noise,
        generator=source_generator,
    )
    target = make_toy_samples(
        target_centers,
        args.sample_count,
        noise_scale=args.cluster_noise,
        generator=target_generator,
    )
    ot_result = sinkhorn(
        source,
        target,
        epsilon=args.ot_epsilon,
        max_iterations=args.ot_iterations,
        cost="sqeuclidean",
    )
    pair_generator = torch.Generator().manual_seed(args.seed + 2)
    paired_endpoints = sample_coupling(
        ot_result.plan,
        source,
        target,
        args.paired_samples,
        generator=pair_generator,
    )
    model, history = train_bidirectional_model(
        paired_endpoints,
        ot_result.plan,
        source,
        target,
        batch_size=args.batch_size,
        candidate_count=args.candidates,
        hidden_width=args.hidden_width,
        learning_rate=args.learning_rate,
        steps=args.steps,
        seed=args.seed + 3,
        device=device,
    )
    forward_samples = sample_direction(
        model,
        source,
        direction=1,
        seed=args.seed + 4,
        device=device,
    )
    backward_samples = sample_direction(
        model,
        target,
        direction=-1,
        seed=args.seed + 5,
        device=device,
    )
    forward_result = summarize_transport(
        forward_samples,
        target,
        epsilon=args.ot_epsilon,
        iterations=args.ot_iterations,
    )
    backward_result = summarize_transport(
        backward_samples,
        source,
        epsilon=args.ot_epsilon,
        iterations=args.ot_iterations,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(
        source,
        target,
        paired_endpoints,
        forward_samples,
        backward_samples,
        history,
        args.output_path,
    )
    summary = {
        "device": str(device),
        "ot_transport_cost": float(ot_result.transport_cost),
        "forward_transport_cost": float(forward_result.transport_cost),
        "backward_transport_cost": float(backward_result.transport_cost),
        "final_loss": history[-1],
        "candidate_count": args.candidates,
        "steps": args.steps,
        "output_path": str(args.output_path),
    }
    summary_path = args.output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
