"""Render Gaussian-to-dog and Gaussian-to-cat RAE latent OT-SDE paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from render_rae_ot_sde_roundtrip import (
    decode_latents,
    encode_pool,
    load_rae,
    save_contact_sheet,
    select_paths,
    tensor_to_image,
)
from torch import Tensor, nn

from brown_bridge.ot import SinkhornResult, sinkhorn_from_cost
from brown_bridge.spatial_stochastic import sample_pinned_brownian_bridge


def mean_squared_cost(source: Tensor, target: Tensor) -> Tensor:
    source_flat = source.flatten(1).double().cpu()
    target_flat = target.flatten(1).double().cpu()
    dimension = source_flat.shape[1]
    cost = (
        source_flat.square().sum(1, keepdim=True)
        + target_flat.square().sum(1)[None, :]
        - 2.0 * source_flat @ target_flat.T
    ) / dimension
    return cost.clamp_min(0.0)


def solve_domain_ot(
    gaussian: Tensor,
    target: Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> SinkhornResult:
    result = sinkhorn_from_cost(
        mean_squared_cost(gaussian, target),
        epsilon=epsilon,
        max_iterations=iterations,
        tolerance=1e-8,
    )
    if not result.converged:
        raise RuntimeError(
            "Sinkhorn did not reach the requested marginal tolerance: "
            f"row={float(result.row_marginal_error):.3g}, "
            f"column={float(result.column_marginal_error):.3g}"
        )
    return result


def sample_conditional_target(
    plan: Tensor,
    source_index: int,
    *,
    seed: int,
) -> int:
    conditional = plan[source_index].float()
    conditional = conditional / conditional.sum()
    generator = torch.Generator().manual_seed(seed)
    return int(torch.multinomial(conditional, 1, generator=generator))


def sample_domain_path(
    gaussian: Tensor,
    target: Tensor,
    noise: Tensor,
    *,
    epsilon: float,
    steps: int,
    microsteps: int,
) -> Tensor:
    return sample_pinned_brownian_bridge(
        gaussian.flatten(1),
        target.flatten(1),
        epsilon=epsilon,
        macro_steps=steps,
        microsteps_per_macro=microsteps,
        noise=noise,
    ).reshape(steps + 1, *gaussian.shape)[:, 0]


def save_path_gif(
    decoded: Tensor,
    output: Path,
    *,
    frame_duration_ms: int,
) -> None:
    frames: list[Image.Image] = [tensor_to_image(value) for value in decoded]
    durations = [frame_duration_ms] * len(frames)
    durations[0] = 3 * frame_duration_ms
    durations[-1] = 3 * frame_duration_ms
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def ot_metrics(
    result: SinkhornResult,
    source_index: int,
    target_index: int,
) -> dict[str, float | int | bool]:
    return {
        "transport_cost": float(result.transport_cost),
        "regularized_objective": float(result.regularized_objective),
        "selected_pair_cost": float(result.cost_matrix[source_index, target_index]),
        "selected_pair_mass": float(result.plan[source_index, target_index]),
        "row_marginal_error": float(result.row_marginal_error),
        "column_marginal_error": float(result.column_marginal_error),
        "iterations": result.iterations,
        "converged": result.converged,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this renderer requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model: nn.Module = load_rae(args.rae_config, device)

    domain_paths = {
        domain: select_paths(
            args.afhq_root,
            domain,
            split=args.split,
            pool_size=args.pool_size,
            seed=args.seed + domain_index + 1,
        )
        for domain_index, domain in enumerate(("dog", "cat"))
    }
    target_latents = {
        domain: encode_pool(
            model,
            paths,
            image_size=args.image_size,
            batch_size=args.encode_batch_size,
            device=device,
        )
        for domain, paths in domain_paths.items()
    }
    latent_shape = target_latents["dog"].shape[1:]
    gaussian_generator = torch.Generator().manual_seed(args.seed + 10)
    gaussian = torch.randn(
        (args.pool_size, *latent_shape),
        generator=gaussian_generator,
    )
    source_index = int(
        torch.randint(
            args.pool_size,
            (1,),
            generator=torch.Generator().manual_seed(args.seed + 11),
        )
    )
    plans = {
        domain: solve_domain_ot(
            gaussian,
            target,
            epsilon=args.ot_epsilon,
            iterations=args.ot_iterations,
        )
        for domain, target in target_latents.items()
    }
    target_indices = {
        domain: sample_conditional_target(
            plans[domain].plan,
            source_index,
            seed=args.seed + domain_index + 20,
        )
        for domain_index, domain in enumerate(("dog", "cat"))
    }

    selected_gaussian = gaussian[source_index : source_index + 1].to(device)
    total_steps = args.steps * args.microsteps
    bridge_noise = torch.randn(
        (total_steps - 1, 1, selected_gaussian.numel()),
        generator=torch.Generator().manual_seed(args.seed + 30),
    ).to(device)
    latent_paths = {
        domain: sample_domain_path(
            selected_gaussian,
            target_latents[domain][
                target_indices[domain] : target_indices[domain] + 1
            ].to(device),
            bridge_noise,
            epsilon=args.bridge_epsilon,
            steps=args.steps,
            microsteps=args.microsteps,
        )
        for domain in ("dog", "cat")
    }
    decoded_paths = {
        domain: decode_latents(
            model,
            latent_path,
            batch_size=args.decode_batch_size,
            device=device,
        )
        for domain, latent_path in latent_paths.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for domain in ("dog", "cat"):
        gif_path = args.output_dir / f"gaussian_to_{domain}_rae_ot_sde.gif"
        sheet_path = (
            args.output_dir / f"gaussian_to_{domain}_rae_ot_sde_contact_sheet.png"
        )
        save_path_gif(
            decoded_paths[domain],
            gif_path,
            frame_duration_ms=args.frame_duration_ms,
        )
        save_contact_sheet(decoded_paths[domain], sheet_path)
        outputs[f"{domain}_gif"] = str(gif_path)
        outputs[f"{domain}_contact_sheet"] = str(sheet_path)

    summary: dict[str, object] = {
        "method": "empirical Gaussian-to-domain balanced OT in standardized RAE latent MSE geometry plus exact conditional pinned Brownian SDEs",
        "claim_limit": "endpoint-conditioned empirical bridges; not a learned source-only Gaussian-to-class generator",
        "decoder": "official pretrained RAE DINOv2-B ViT-XL decoder",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "pool_size": args.pool_size,
        "latent_shape": list(latent_shape),
        "shared_gaussian_source_index": source_index,
        "shared_brownian_noise": True,
        "steps": args.steps,
        "microsteps": args.microsteps,
        "ot_epsilon": args.ot_epsilon,
        "bridge_epsilon": args.bridge_epsilon,
        "selected_targets": {
            domain: str(domain_paths[domain][target_indices[domain]])
            for domain in ("dog", "cat")
        },
        "ot": {
            domain: ot_metrics(plans[domain], source_index, target_indices[domain])
            for domain in ("dog", "cat")
        },
        "outputs": outputs,
    }
    (args.output_dir / "gaussian_domain_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rae-config", type=Path, required=True)
    parser.add_argument("--afhq-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iterations", type=int, default=1000)
    parser.add_argument("--bridge-epsilon", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--frame-duration-ms", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    positive_values = {
        "image size": args.image_size,
        "pool size": args.pool_size,
        "encode batch size": args.encode_batch_size,
        "decode batch size": args.decode_batch_size,
        "OT epsilon": args.ot_epsilon,
        "OT iterations": args.ot_iterations,
        "steps": args.steps,
        "microsteps": args.microsteps,
        "frame duration": args.frame_duration_ms,
    }
    invalid_names = [name for name, value in positive_values.items() if value <= 0]
    if invalid_names or args.bridge_epsilon < 0:
        parser.error(
            "expected positive numeric settings and nonnegative bridge epsilon; "
            f"invalid: {invalid_names}"
        )
    return args


if __name__ == "__main__":
    run(parse_args())
