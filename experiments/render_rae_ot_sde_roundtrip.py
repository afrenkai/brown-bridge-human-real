"""Render a reciprocal RAE-latent OT Brownian bridge as a round-trip GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor, nn
from torchvision.transforms.functional import pil_to_tensor
from utils.model_utils import instantiate_from_config

from brown_bridge.bridges import sample_pinned_brownian_bridge
from brown_bridge.features import (
    compact_spatial_content_descriptor,
    content_cost_matrix,
    species_direction,
)
from brown_bridge.ot import sinkhorn_from_cost


def load_image(path: Path, image_size: int) -> Tensor:
    with Image.open(path) as opened:
        image = ImageOps.fit(
            opened.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
        )
    return pil_to_tensor(image).float().div(255.0)


def select_paths(
    root: Path,
    domain: str,
    *,
    split: str,
    pool_size: int,
    seed: int,
) -> list[Path]:
    candidates = sorted((root / split / domain).glob("*.jpg"))
    if len(candidates) < pool_size:
        raise FileNotFoundError(
            f"expected at least {pool_size} AFHQ {split} {domain} images under {root}"
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(candidates), generator=generator)[:pool_size]
    return [candidates[int(index)] for index in indices]


def load_rae(config_path: Path, device: torch.device) -> nn.Module:
    configuration = OmegaConf.load(config_path)
    if "stage_1" not in configuration:
        raise ValueError(f"{config_path} has no stage_1 configuration")
    model: nn.Module = instantiate_from_config(configuration["stage_1"])
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def encode_pool(
    model: nn.Module,
    paths: list[Path],
    *,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    encoded_batches: list[Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([load_image(path, image_size) for path in batch_paths])
        encoded_batches.append(model.encode(images.to(device)).float().cpu())
    return torch.cat(encoded_batches)


def unnormalize_latent(model: nn.Module, latent: Tensor) -> Tensor:
    if not model.do_normalization:
        return latent.float().cpu()
    mean = 0.0 if model.latent_mean is None else model.latent_mean.float().cpu()
    variance = 1.0 if model.latent_var is None else model.latent_var.float().cpu()
    return latent.float().cpu() * torch.sqrt(variance + model.eps) + mean


def rae_content_descriptor(
    dog_latent: Tensor,
    cat_latent: Tensor,
    model: nn.Module,
) -> tuple[Tensor, Tensor]:
    raw_dog = unnormalize_latent(model, dog_latent)
    raw_cat = unnormalize_latent(model, cat_latent)
    dog_global = F.normalize(raw_dog.mean(dim=(2, 3)), dim=1)
    cat_global = F.normalize(raw_cat.mean(dim=(2, 3)), dim=1)
    direction = species_direction(dog_global, cat_global)
    dog_patches = raw_dog.flatten(2).transpose(1, 2)
    cat_patches = raw_cat.flatten(2).transpose(1, 2)
    dog_descriptor = compact_spatial_content_descriptor(
        dog_global, dog_patches, direction
    )
    cat_descriptor = compact_spatial_content_descriptor(
        cat_global, cat_patches, direction
    )
    return dog_descriptor, cat_descriptor


def select_ot_pair(
    dog_descriptor: Tensor,
    cat_descriptor: Tensor,
    *,
    epsilon: float,
    iterations: int,
    seed: int,
) -> tuple[int, int, dict[str, float | int | bool]]:
    cost = content_cost_matrix(dog_descriptor, cat_descriptor)
    result = sinkhorn_from_cost(
        cost,
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
    generator = torch.Generator().manual_seed(seed)
    flat_index = int(
        torch.multinomial(
            result.plan.flatten().float(),
            1,
            replacement=True,
            generator=generator,
        )
    )
    dog_index, cat_index = divmod(flat_index, result.plan.shape[1])
    metrics: dict[str, float | int | bool] = {
        "transport_cost": float(result.transport_cost),
        "regularized_objective": float(result.regularized_objective),
        "selected_pair_cost": float(result.cost_matrix[dog_index, cat_index]),
        "selected_pair_mass": float(result.plan[dog_index, cat_index]),
        "row_marginal_error": float(result.row_marginal_error),
        "column_marginal_error": float(result.column_marginal_error),
        "iterations": result.iterations,
        "converged": result.converged,
    }
    return dog_index, cat_index, metrics


def roundtrip_bridge(
    dog_latent: Tensor,
    cat_latent: Tensor,
    *,
    epsilon: float,
    half_steps: int,
    microsteps: int,
) -> Tensor:
    dog_to_cat = sample_pinned_brownian_bridge(
        dog_latent.flatten(1),
        cat_latent.flatten(1),
        epsilon=epsilon,
        macro_steps=half_steps,
        microsteps_per_macro=microsteps,
    ).reshape(half_steps + 1, *dog_latent.shape)
    cat_to_dog = torch.flip(dog_to_cat, dims=(0,))
    return torch.cat([dog_to_cat, cat_to_dog[1:]], dim=0)


@torch.inference_mode()
def decode_latents(
    model: nn.Module,
    latent: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    decoded_batches: list[Tensor] = []
    for start in range(0, len(latent), batch_size):
        decoded = model.decode(latent[start : start + batch_size].to(device))
        decoded_batches.append(decoded.float().clamp(0.0, 1.0).cpu())
    return torch.cat(decoded_batches)


def tensor_to_image(value: Tensor) -> Image.Image:
    array = value.clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_roundtrip_gif(
    decoded: Tensor,
    output: Path,
    *,
    frame_duration_ms: int,
) -> None:
    midpoint = (len(decoded) - 1) // 2
    frames = [tensor_to_image(value) for value in decoded]
    durations = [frame_duration_ms] * len(frames)
    durations[0] = 3 * frame_duration_ms
    durations[midpoint] = 3 * frame_duration_ms
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


def save_contact_sheet(decoded: Tensor, output: Path) -> None:
    indices = torch.linspace(0, len(decoded) - 1, 9).round().long().tolist()
    images = [tensor_to_image(decoded[index]) for index in indices]
    tile_size = images[0].width
    header_height = 32
    canvas = Image.new(
        "RGB", (len(images) * tile_size, tile_size + header_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, image in enumerate(images):
        draw.text((column * tile_size + 8, 10), f"t={column / 8:.3g}", fill="black")
        canvas.paste(image, (column * tile_size, header_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this renderer requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = load_rae(args.rae_config, device)

    dog_paths = select_paths(
        args.afhq_root,
        "dog",
        split=args.split,
        pool_size=args.pool_size,
        seed=args.seed + 1,
    )
    cat_paths = select_paths(
        args.afhq_root,
        "cat",
        split=args.split,
        pool_size=args.pool_size,
        seed=args.seed + 2,
    )
    dog_latent = encode_pool(
        model,
        dog_paths,
        image_size=args.image_size,
        batch_size=args.encode_batch_size,
        device=device,
    )
    cat_latent = encode_pool(
        model,
        cat_paths,
        image_size=args.image_size,
        batch_size=args.encode_batch_size,
        device=device,
    )
    dog_descriptor, cat_descriptor = rae_content_descriptor(
        dog_latent, cat_latent, model
    )
    dog_index, cat_index, ot_metrics = select_ot_pair(
        dog_descriptor,
        cat_descriptor,
        epsilon=args.ot_epsilon,
        iterations=args.ot_iterations,
        seed=args.seed + 3,
    )
    selected_dog = dog_latent[dog_index : dog_index + 1].to(device)
    selected_cat = cat_latent[cat_index : cat_index + 1].to(device)
    latent_roundtrip = roundtrip_bridge(
        selected_dog,
        selected_cat,
        epsilon=args.bridge_epsilon,
        half_steps=args.half_steps,
        microsteps=args.microsteps,
    )[:, 0]
    decoded = decode_latents(
        model,
        latent_roundtrip,
        batch_size=args.decode_batch_size,
        device=device,
    )

    gif_path = args.output_dir / "dog_cat_dog_rae_ot_sde.gif"
    sheet_path = args.output_dir / "dog_cat_dog_rae_ot_sde_contact_sheet.png"
    save_roundtrip_gif(
        decoded,
        gif_path,
        frame_duration_ms=args.frame_duration_ms,
    )
    save_contact_sheet(decoded, sheet_path)
    summary: dict[str, object] = {
        "method": "balanced entropic OT descriptor coupling plus an exact pinned Brownian bridge in standardized frozen RAE latents",
        "geometry_note": "the species-projected OT descriptor cost and isotropic standardized-latent Brownian reference are different geometries",
        "endpoint_selection": "one seeded categorical draw from the balanced entropic OT plan",
        "roundtrip": "the cat-to-dog half is the exact time reversal of the sampled dog-to-cat realization",
        "decoder": "official pretrained RAE DINOv2-B ViT-XL decoder",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "pool_size_per_domain": args.pool_size,
        "selected_dog": str(dog_paths[dog_index]),
        "selected_cat": str(cat_paths[cat_index]),
        "latent_shape": list(selected_dog.shape[1:]),
        "frames": len(decoded),
        "ot": ot_metrics,
        "ot_epsilon": args.ot_epsilon,
        "bridge_epsilon": args.bridge_epsilon,
        "half_steps": args.half_steps,
        "microsteps": args.microsteps,
        "gif": str(gif_path),
        "contact_sheet": str(sheet_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    parser.add_argument("--half-steps", type=int, default=20)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--frame-duration-ms", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    positive_values = {
        "image size": args.image_size,
        "pool size": args.pool_size,
        "encode batch size": args.encode_batch_size,
        "decode batch size": args.decode_batch_size,
        "OT epsilon": args.ot_epsilon,
        "OT iterations": args.ot_iterations,
        "half steps": args.half_steps,
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
