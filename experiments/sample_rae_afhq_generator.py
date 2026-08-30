from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import torch
from omegaconf import OmegaConf
from PIL import Image
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import Sampler, create_transport
from torchvision.io import write_png
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-per-class", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--ode-steps", type=int, default=21)
    parser.add_argument("--sde-steps", type=int, default=41)
    parser.add_argument("--skip-trajectories", action="store_true")
    parser.add_argument("--writer-workers", type=int, default=8)
    return parser.parse_args()


def load_models(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    RAE,
    Stage2ModelProtocol,
    dict[str, object],
    dict[str, object],
]:
    config = OmegaConf.load(config_path)
    config.stage_2.ckpt = str(checkpoint_path)
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
    del sampler_config, guidance_config, training_config, evaluation_config
    rae = instantiate_from_config(rae_config).to(device).eval()
    model = instantiate_from_config(model_config).to(device).eval()
    transport_values = dict(transport_config.get("params", {}))
    misc_values = dict(misc_config)
    return rae, model, transport_values, misc_values


def build_sampler(
    transport_values: dict[str, object],
    misc_values: dict[str, object],
    mode: str,
    steps: int,
) -> object:
    latent_size = tuple(int(dimension) for dimension in misc_values["latent_size"])
    shift_dimension = int(
        misc_values.get("time_dist_shift_dim", math.prod(latent_size))
    )
    shift_base = int(misc_values.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(shift_dimension / shift_base)
    transport = create_transport(**transport_values, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    if mode == "ode":
        return sampler.sample_ode(
            sampling_method="euler",
            num_steps=steps,
            atol=1.0e-6,
            rtol=1.0e-3,
            reverse=False,
        )
    if mode == "sde":
        return sampler.sample_sde(
            sampling_method="euler",
            diffusion_form="sigma",
            diffusion_norm=1.0,
            last_step=None,
            last_step_size=0.0,
            num_steps=steps,
        )
    raise ValueError(f"Unsupported sampler mode: {mode}")


def run_sampler(
    sample_function: object,
    model: Stage2ModelProtocol,
    noise: torch.Tensor,
    labels: torch.Tensor,
    cfg_scale: float,
    null_label: int,
) -> torch.Tensor | list[torch.Tensor]:
    if cfg_scale <= 1.0:
        return sample_function(noise, model.forward, y=labels)
    sample_noise = torch.cat((noise, noise), dim=0)
    null_labels = torch.full_like(labels, null_label)
    sample_labels = torch.cat((labels, null_labels), dim=0)
    trajectory = sample_function(
        sample_noise,
        model.forward_with_cfg,
        y=sample_labels,
        cfg_scale=cfg_scale,
        cfg_interval=(0.0, 1.0),
    )
    if isinstance(trajectory, list):
        return [states.chunk(2, dim=0)[0] for states in trajectory]
    return trajectory.chunk(2, dim=1)[0]


def final_latents(trajectory: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(trajectory, list):
        final = trajectory[-1]
    else:
        final = trajectory[-1]
    if not torch.isfinite(final).all():
        raise FloatingPointError("Sampler produced nonfinite final latents")
    return final


def combine_trajectories(
    first: torch.Tensor | list[torch.Tensor],
    second: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(first, list) and isinstance(second, list):
        return [
            torch.cat((first_state, second_state), dim=0)
            for first_state, second_state in zip(first, second, strict=True)
        ]
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return torch.cat((first, second), dim=1)
    raise TypeError("Trajectory containers do not match")


def image_from_tensor(image: torch.Tensor) -> Image.Image:
    pixels = image.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
    array = pixels.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_tensor_png(pixels: torch.Tensor, path: Path) -> None:
    write_png(pixels, str(path), compression_level=1)


def limit_pending_writes(
    pending: list[Future[None]],
    maximum_pending: int,
) -> None:
    while len(pending) >= maximum_pending:
        completed = pending.pop(0)
        completed.result()


def save_grid(images: list[Image.Image], columns: int, path: Path) -> None:
    if not images:
        return
    rows = math.ceil(len(images) / columns)
    width, height = images[0].size
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        x_coordinate = index % columns
        y_coordinate = index // columns
        grid.paste(image, (x_coordinate * width, y_coordinate * height))
    grid.save(path)


def save_gif(frames: list[Image.Image], path: Path) -> None:
    if not frames:
        raise ValueError("A GIF requires at least one frame")
    durations = (
        [300] + [80] * max(0, len(frames) - 2) + ([500] if len(frames) > 1 else [])
    )
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )


def decode_trajectory(
    rae: RAE,
    trajectory: torch.Tensor | list[torch.Tensor],
) -> list[list[Image.Image]]:
    states = trajectory if isinstance(trajectory, list) else list(trajectory.unbind(0))
    decoded_by_sample: list[list[Image.Image]] = []
    for state in states:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoded = rae.decode(state).clamp(0, 1)
        if not decoded_by_sample:
            decoded_by_sample = [[] for sample_index in range(decoded.shape[0])]
        for sample_index, image in enumerate(decoded):
            decoded_by_sample[sample_index].append(image_from_tensor(image))
    return decoded_by_sample


def render_trajectories(
    rae: RAE,
    model: Stage2ModelProtocol,
    transport_values: dict[str, object],
    misc_values: dict[str, object],
    output_dir: Path,
    generator: torch.Generator,
    cfg_scale: float,
    ode_steps: int,
    sde_steps: int,
) -> None:
    latent_size = tuple(int(dimension) for dimension in misc_values["latent_size"])
    null_label = int(misc_values.get("null_label", 1000))
    shared_noise = torch.randn((1, *latent_size), device="cuda", generator=generator)
    noise = shared_noise.repeat(2, 1, 1, 1)
    labels = torch.tensor((0, 1), device="cuda", dtype=torch.long)
    for mode, steps in (("ode", ode_steps), ("sde", sde_steps)):
        sample_function = build_sampler(transport_values, misc_values, mode, steps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if mode == "sde":
                random_state = torch.cuda.get_rng_state()
                cat_trajectory = run_sampler(
                    sample_function,
                    model,
                    noise[:1],
                    labels[:1],
                    cfg_scale,
                    null_label,
                )
                torch.cuda.set_rng_state(random_state)
                dog_trajectory = run_sampler(
                    sample_function,
                    model,
                    noise[1:],
                    labels[1:],
                    cfg_scale,
                    null_label,
                )
                trajectory = combine_trajectories(
                    cat_trajectory,
                    dog_trajectory,
                )
                if not isinstance(trajectory, list):
                    raise TypeError("SDE trajectory must be a list")
                trajectory = [noise] + trajectory[:-1]
            else:
                trajectory = run_sampler(
                    sample_function,
                    model,
                    noise,
                    labels,
                    cfg_scale,
                    null_label,
                )
        final_latents(trajectory)
        decoded = decode_trajectory(rae, trajectory)
        class_names = ("cat", "dog")
        for class_name, frames in zip(class_names, decoded, strict=True):
            save_gif(frames, output_dir / f"gaussian_to_{class_name}_{mode}.gif")
        paired_frames: list[Image.Image] = []
        for cat_frame, dog_frame in zip(decoded[0], decoded[1], strict=True):
            paired = Image.new(
                "RGB", (cat_frame.width + dog_frame.width, cat_frame.height)
            )
            paired.paste(cat_frame, (0, 0))
            paired.paste(dog_frame, (cat_frame.width, 0))
            paired_frames.append(paired)
        save_gif(
            paired_frames, output_dir / f"gaussian_to_cat_dog_same_noise_{mode}.gif"
        )


def generate_images(
    rae: RAE,
    model: Stage2ModelProtocol,
    transport_values: dict[str, object],
    misc_values: dict[str, object],
    output_dir: Path,
    generator: torch.Generator,
    num_per_class: int,
    batch_size: int,
    cfg_scale: float,
    ode_steps: int,
    writer_workers: int,
) -> list[dict[str, object]]:
    sample_function = build_sampler(transport_values, misc_values, "ode", ode_steps)
    latent_size = tuple(int(dimension) for dimension in misc_values["latent_size"])
    null_label = int(misc_values.get("null_label", 1000))
    records: list[dict[str, object]] = []
    grid_images: dict[str, list[Image.Image]] = {"cat": [], "dog": []}
    pending_writes: list[Future[None]] = []
    writer = ThreadPoolExecutor(max_workers=writer_workers)
    for class_label, class_name in ((0, "cat"), (1, "dog")):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        generated_count = 0
        while generated_count < num_per_class:
            current_batch_size = min(batch_size, num_per_class - generated_count)
            noise = torch.randn(
                (current_batch_size, *latent_size),
                device="cuda",
                generator=generator,
            )
            labels = torch.full(
                (current_batch_size,),
                class_label,
                device="cuda",
                dtype=torch.long,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                trajectory = run_sampler(
                    sample_function,
                    model,
                    noise,
                    labels,
                    cfg_scale,
                    null_label,
                )
                decoded = rae.decode(final_latents(trajectory)).clamp(0, 1)
            for batch_index, decoded_image in enumerate(decoded):
                image_index = generated_count + batch_index
                relative_path = Path(class_name) / f"{image_index:06d}.png"
                pixels = (
                    decoded_image.detach()
                    .float()
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .to(device="cpu", dtype=torch.uint8)
                )
                pending_writes.append(
                    writer.submit(save_tensor_png, pixels, output_dir / relative_path)
                )
                limit_pending_writes(pending_writes, writer_workers * 8)
                if len(grid_images[class_name]) < 16:
                    array = pixels.permute(1, 2, 0).numpy()
                    grid_images[class_name].append(Image.fromarray(array))
                records.append(
                    {
                        "class_name": class_name,
                        "class_label": class_label,
                        "index": image_index,
                        "path": str(relative_path),
                    }
                )
            generated_count += current_batch_size
    for pending_write in pending_writes:
        pending_write.result()
    writer.shutdown()
    for class_name, images in grid_images.items():
        save_grid(images, 4, output_dir / f"{class_name}_grid.png")
    return records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_per_class < 1 or args.batch_size < 1 or args.writer_workers < 1:
        raise ValueError(
            "Sample counts, batch size, and writer workers must be positive"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started_at = perf_counter()
    rae, model, transport_values, misc_values = load_models(
        args.config,
        args.checkpoint,
        device,
    )
    if not args.skip_trajectories:
        render_trajectories(
            rae,
            model,
            transport_values,
            misc_values,
            args.output_dir,
            generator,
            args.cfg_scale,
            args.ode_steps,
            args.sde_steps,
        )
    records = generate_images(
        rae,
        model,
        transport_values,
        misc_values,
        args.output_dir,
        generator,
        args.num_per_class,
        args.batch_size,
        args.cfg_scale,
        args.ode_steps,
        args.writer_workers,
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "ode_steps": args.ode_steps,
        "sde_steps": args.sde_steps,
        "num_per_class": args.num_per_class,
        "elapsed_seconds": perf_counter() - started_at,
        "samples": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "samples"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
