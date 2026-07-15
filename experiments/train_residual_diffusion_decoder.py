"""Train and apply a compact AFHQ residual-diffusion decoder on Apple MPS."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms.functional import gaussian_blur

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.continuous_vae import ContinuousConditionalVAE
from brown_bridge.residual_diffusion import (
    ResidualDenoiser,
    cosine_alpha_bar,
    ddim_sample,
    q_sample,
    refine_image,
)
from render_dino_stargan_bridge import make_distribution_gif, tensor_to_pil
from train_continuous_vae_renderer import AFHQDomainDataset, image_edges


def load_vae(path: Path, device: torch.device) -> ContinuousConditionalVAE:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ContinuousConditionalVAE(
        image_size=128,
        global_dim=192,
        patch_dim=192,
        patch_grid=4,
        latent_grid=8,
        latent_channels=32,
        condition_dim=128,
        base_channels=32,
    )
    model.conditioning_scale = 0.0
    model.load_state_dict(payload["model"])
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def reconstruct(model: ContinuousConditionalVAE, images: Tensor) -> Tensor:
    mean, _ = model.encode(images)
    batch = len(images)
    return model.decode(
        mean,
        torch.zeros(batch, device=images.device),
        torch.zeros(batch, 192, device=images.device),
        torch.zeros(batch, 16, 192, device=images.device),
    )


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_value, value in zip(ema.parameters(), model.parameters()):
            ema_value.lerp_(value, 1.0 - decay)
        for ema_value, value in zip(ema.buffers(), model.buffers()):
            ema_value.copy_(value)


def predicted_clean(
    noisy: Tensor, predicted_noise: Tensor, time_index: Tensor, alpha_bar: Tensor
) -> Tensor:
    cumulative = alpha_bar[time_index].to(noisy)
    shape = (len(noisy),) + (1,) * (noisy.ndim - 1)
    cumulative = cumulative.reshape(shape)
    return (noisy - (1.0 - cumulative).sqrt() * predicted_noise) / cumulative.sqrt()


def regularizers(clean_prediction: Tensor, reliable: Tensor) -> tuple[Tensor, Tensor]:
    """Penalize residual color drift and isolated over-range impulses."""
    if not reliable.any():
        zero = clean_prediction.new_zeros(())
        return zero, zero
    selected = clean_prediction[reliable]
    low_frequency = F.avg_pool2d(selected, kernel_size=9, stride=1, padding=4).square().mean()
    over_range = F.relu(selected.abs() - 1.0).square().mean()
    return low_frequency, over_range


@torch.no_grad()
def validation_metrics(
    vae: ContinuousConditionalVAE,
    diffusion: ResidualDenoiser,
    loader: DataLoader[Tensor],
    alpha_bar: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    images = next(iter(loader)).to(device)
    base = reconstruct(vae, images)
    generator = torch.Generator().manual_seed(args.seed + 91)
    noise = torch.randn(base.shape, generator=generator).to(device)
    if args.zero_initial_noise:
        noise.zero_()
    residual = ddim_sample(
        diffusion,
        base.mul(2).sub(1),
        alpha_bar,
        sampling_steps=args.sampling_steps,
        initial_noise=noise,
        start_fraction=args.sampling_strength,
    )
    if args.residual_blur_sigma > 0:
        residual = gaussian_blur(residual, kernel_size=7, sigma=args.residual_blur_sigma)
    refined = refine_image(base, residual, residual_scale=args.residual_scale)
    base_dx, base_dy = image_edges(base)
    refined_dx, refined_dy = image_edges(refined)
    image_dx, image_dy = image_edges(images)
    return {
        "base_l1": float(F.l1_loss(base, images)),
        "refined_l1": float(F.l1_loss(refined, images)),
        "base_edge_l1": float(
            0.5 * (F.l1_loss(base_dx, image_dx) + F.l1_loss(base_dy, image_dy))
        ),
        "refined_edge_l1": float(
            0.5 * (F.l1_loss(refined_dx, image_dx) + F.l1_loss(refined_dy, image_dy))
        ),
        "base_edge_energy": float(0.5 * (base_dx.abs().mean() + base_dy.abs().mean())),
        "refined_edge_energy": float(
            0.5 * (refined_dx.abs().mean() + refined_dy.abs().mean())
        ),
        "real_edge_energy": float(0.5 * (image_dx.abs().mean() + image_dy.abs().mean())),
    }


def train(args: argparse.Namespace, device: torch.device) -> tuple[ResidualDenoiser, dict[str, object]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    vae = load_vae(args.vae_checkpoint, device)
    train_data = ConcatDataset(
        [
            AFHQDomainDataset(args.afhq_root, domain, image_size=128, training=True)
            for domain in ("cat", "dog")
        ]
    )
    test_data = ConcatDataset(
        [
            AFHQDomainDataset(args.afhq_root, domain, image_size=128, training=False)
            for domain in ("cat", "dog")
        ]
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    validation_loader = DataLoader(test_data, batch_size=args.validation_batch, shuffle=True)
    model = ResidualDenoiser(base_channels=args.base_channels).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    alpha_bar = cosine_alpha_bar(args.diffusion_steps, device=device)
    iterator = iter(loader)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, args.steps + 1):
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images = next(iterator)
        images = images.to(device)
        with torch.no_grad():
            base = reconstruct(vae, images)
            clean = ((images - base) / args.residual_scale).clamp(-1.25, 1.25)
        time_index = torch.randint(args.diffusion_steps, (len(images),), device=device)
        noise = torch.randn_like(clean)
        noisy = q_sample(clean, noise, time_index, alpha_bar)
        predicted_noise = model(
            noisy, base.mul(2).sub(1), time_index.float() / (args.diffusion_steps - 1)
        )
        denoising = F.mse_loss(predicted_noise, noise)
        clean_prediction = predicted_clean(noisy, predicted_noise, time_index, alpha_bar)
        low_frequency, over_range = regularizers(
            clean_prediction, alpha_bar[time_index] >= args.regularizer_min_alpha
        )
        loss = (
            denoising
            + args.lambda_low_frequency * low_frequency
            + args.lambda_over_range * over_range
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema, model, args.ema_decay)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "denoising": float(denoising.detach()),
                "low_frequency": float(low_frequency.detach()),
                "over_range": float(over_range.detach()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    metrics = validation_metrics(
        vae, ema, validation_loader, alpha_bar, args, device
    )
    summary: dict[str, object] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_images": len(train_data),
        "test_images": len(test_data),
        "history": history,
        "validation": metrics,
    }
    return ema, summary


def make_comparison(base: Tensor, refined: Tensor, output: Path) -> None:
    rows, steps = base.shape[:2]
    tile, header, label_width = 192, 34, 105
    canvas = Image.new("RGB", (label_width + steps * tile, header + rows * tile * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for step in range(steps):
        draw.text((label_width + step * tile + 8, 11), f"t={step/(steps-1):.2f}", fill="black")
    for row in range(rows):
        for variant, (name, trajectory) in enumerate((("VAE", base), ("+ residual SDE", refined))):
            y = header + (2 * row + variant) * tile
            draw.text((5, y + 84), name, fill="black")
            for step in range(steps):
                image = tensor_to_pil(trajectory[row, step].mul(2).sub(1))
                canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (label_width + step * tile, y))
    canvas.save(output)


@torch.no_grad()
def render_trajectories(
    model: ResidualDenoiser,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    payload = torch.load(args.trajectory_file, map_location="cpu", weights_only=False)
    base = payload["trajectory_128"].to(device)
    rows, steps = base.shape[:2]
    generator = torch.Generator().manual_seed(args.seed + 117)
    path_noise = torch.randn((rows, 1, *base.shape[2:]), generator=generator)
    if args.zero_initial_noise:
        path_noise.zero_()
    shared_noise = path_noise.expand(rows, steps, *base.shape[2:]).reshape(-1, *base.shape[2:]).to(device)
    alpha_bar = cosine_alpha_bar(args.diffusion_steps, device=device)
    residual = ddim_sample(
        model,
        base.flatten(0, 1).mul(2).sub(1),
        alpha_bar,
        sampling_steps=args.sampling_steps,
        initial_noise=shared_noise,
        start_fraction=args.sampling_strength,
    ).reshape_as(base)
    if args.residual_blur_sigma > 0:
        residual = gaussian_blur(
            residual.flatten(0, 1), kernel_size=7, sigma=args.residual_blur_sigma
        ).reshape_as(base)
    refined = refine_image(base, residual, residual_scale=args.residual_scale)
    refined[:, 0] = base[:, 0]
    refined[:, -1] = base[:, -1]
    refined_256 = F.interpolate(
        refined.flatten(0, 1), size=(256, 256), mode="bicubic", align_corners=False
    ).clamp(0, 1).reshape(rows, steps, 3, 256, 256)
    base_256 = payload["trajectory_256"]
    make_comparison(base_256, refined_256.cpu(), args.output_dir / "vae_vs_residual_diffusion.png")
    paths = [
        [tensor_to_pil(image.mul(2).sub(1)) for image in row]
        for row in refined_256.cpu()
    ]
    make_distribution_gif(
        paths,
        args.output_dir / "residual_diffusion_distribution.gif",
        clock_label="VAE + residual SDE",
    )
    torch.save(
        {"trajectory_128": refined.cpu(), "trajectory_256": refined_256.cpu()},
        args.output_dir / "residual_diffusion_trajectories.pt",
    )
    base_dx, base_dy = image_edges(base[:, 1:-1].flatten(0, 1))
    refined_dx, refined_dy = image_edges(refined[:, 1:-1].flatten(0, 1))
    return {
        "base_interior_edge_energy": float(0.5 * (base_dx.abs().mean() + base_dy.abs().mean())),
        "refined_interior_edge_energy": float(
            0.5 * (refined_dx.abs().mean() + refined_dy.abs().mean())
        ),
        "base_mean_adjacent_l1": float((base[:, 1:] - base[:, :-1]).abs().mean()),
        "refined_mean_adjacent_l1": float(
            (refined[:, 1:] - refined[:, :-1]).abs().mean()
        ),
    }


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=project / "outputs/manifold_vae_renderer_final/continuous_vae_renderer.pt",
    )
    parser.add_argument(
        "--trajectory-file",
        type=Path,
        default=project / "outputs/manifold_vae_renderer_final/continuous_vae_trajectories.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--sampling-strength", type=float, default=0.35)
    parser.add_argument("--residual-blur-sigma", type=float, default=1.25)
    parser.add_argument("--zero-initial-noise", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.02)
    parser.add_argument("--lambda-over-range", type=float, default=0.02)
    parser.add_argument("--regularizer-min-alpha", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    device = torch.device("mps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint is None:
        model, summary = train(args, device)
    else:
        payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model = ResidualDenoiser(base_channels=args.base_channels).to(device).eval()
        model.load_state_dict(payload["model"])
        summary = {"source_checkpoint": str(args.resume_checkpoint)}
    trajectory_metrics = render_trajectories(model, args, device)
    summary["trajectory"] = trajectory_metrics
    checkpoint = {
        "model": model.cpu().state_dict(),
        "summary": summary,
        "config": vars(args),
    }
    torch.save(checkpoint, args.output_dir / "residual_diffusion_decoder.pt")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
