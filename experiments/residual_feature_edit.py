"""Train a source-anchored residual editor from OT-proposed feature pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brown_bridge.editing import (
    SourceConditionedResidualMLP,
    integrate_residual_edit,
    make_residual_edit_batch,
    residual_flow_matching_loss,
)


def device_for_host() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/residual_edit.pt"))
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--macro-steps", type=int, default=10)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifact = torch.load(args.transport, map_location="cpu", weights_only=True)
    source = artifact["cat_features"]
    target = artifact["dog_features"]
    plan = artifact["plan"]
    device = device_for_host()
    generator = torch.Generator().manual_seed(args.seed)
    model = SourceConditionedResidualMLP(source.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []

    model.train()
    for _ in range(args.steps):
        batch = make_residual_edit_batch(
            plan,
            source,
            target,
            args.batch_size,
            geometry="spherical",
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = residual_flow_matching_loss(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    selected = source[: args.examples].to(device)
    times, residuals, edited_features = integrate_residual_edit(
        model,
        selected,
        macro_steps=args.macro_steps,
        microsteps_per_macro=args.microsteps,
        geometry="spherical",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "losses": torch.tensor(losses),
            "source_features": selected.cpu(),
            "times": times.cpu(),
            "residuals": residuals.cpu(),
            "edited_features": edited_features.cpu(),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
