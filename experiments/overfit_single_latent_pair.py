"""Deliberately memorize one quarantined AFHQ cat/dog pair as a capacity check."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.exclusions import load_exclusion_manifest, sha256_file
from brown_bridge.features import species_direction
from brown_bridge.latent_image_bridge import LatentImageAutoencoder
from brown_bridge.models import DinoCatDogClassifier
from render_dino_stargan_bridge import (
    DISPLAY_STEPS,
    dino_encode,
    make_contact_sheet,
    make_distribution_gif,
    tensor_to_pil,
)
from train_continuous_vae_renderer import AFHQDomainDataset, image_edges


def load_tensor(path: Path, *, size: int = 128) -> tuple[Image.Image, Tensor]:
    with Image.open(path) as image:
        full = ImageOps.fit(image.convert("RGB"), (256, 256)).copy()
        small = ImageOps.fit(image.convert("RGB"), (size, size))
        tensor = pil_to_tensor(small).float().div(255.0)
    return full, tensor


def load_autoencoder(args: argparse.Namespace, device: torch.device) -> LatentImageAutoencoder:
    payload = torch.load(args.autoencoder_checkpoint, map_location="cpu", weights_only=False)
    model = LatentImageAutoencoder(args.latent_channels, args.base_channels)
    model.load_state_dict(payload["model"])
    return model.to(device)


def endpoint_loss(prediction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    pixel = F.l1_loss(prediction, target)
    pred_dx, pred_dy = image_edges(prediction)
    target_dx, target_dy = image_edges(target)
    edge = 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))
    pyramid = 0.5 * (
        F.l1_loss(F.avg_pool2d(prediction, 2), F.avg_pool2d(target, 2))
        + F.l1_loss(F.avg_pool2d(prediction, 4), F.avg_pool2d(target, 4))
    )
    return pixel + 0.5 * edge + 0.2 * pyramid, {
        "pixel": pixel,
        "edge": edge,
        "pyramid": pyramid,
    }


@torch.no_grad()
def endpoint_metrics(model: LatentImageAutoencoder, images: Tensor) -> dict[str, float]:
    reconstruction, _ = model(images)
    pred_dx, pred_dy = image_edges(reconstruction)
    image_dx, image_dy = image_edges(images)
    return {
        "l1": float(F.l1_loss(reconstruction, images)),
        "edge_l1": float(
            0.5 * (F.l1_loss(pred_dx, image_dx) + F.l1_loss(pred_dy, image_dy))
        ),
        "max_abs": float((reconstruction - images).abs().max()),
    }


def train(
    model: LatentImageAutoencoder,
    images: Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    device = images.device
    model.eval()
    before = endpoint_metrics(model, images)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    with torch.no_grad():
        teacher_endpoints = teacher.encode(images)
        time = torch.linspace(0.0, 1.0, DISPLAY_STEPS, device=device)
        teacher_latent_path = torch.lerp(
            teacher_endpoints[0:1].expand(DISPLAY_STEPS, -1, -1, -1),
            teacher_endpoints[1:2].expand(DISPLAY_STEPS, -1, -1, -1),
            time[:, None, None, None],
        )
        teacher_path = teacher.decode(teacher_latent_path)
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        endpoints = model.encode(images)
        endpoint_prediction = model.decode(endpoints)
        reconstruction, pieces = endpoint_loss(endpoint_prediction, images)
        path_latent = torch.lerp(
            endpoints[0:1].expand(DISPLAY_STEPS, -1, -1, -1),
            endpoints[1:2].expand(DISPLAY_STEPS, -1, -1, -1),
            time[:, None, None, None],
        )
        path = model.decode(path_latent)
        interior_anchor = F.l1_loss(path[1:-1], teacher_path[1:-1])
        acceleration = (
            path[2:] - 2.0 * path[1:-1] + path[:-2]
        ).abs().mean()
        reencoded = model.encode(path)
        latent_cycle = F.l1_loss(reencoded, path_latent.detach())
        loss = (
            reconstruction
            + args.lambda_interior_anchor * interior_anchor
            + args.lambda_acceleration * acceleration
            + args.lambda_latent_cycle * latent_cycle
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "pixel": float(pieces["pixel"].detach()),
                "edge": float(pieces["edge"].detach()),
                "interior_anchor": float(interior_anchor.detach()),
                "acceleration": float(acceleration.detach()),
                "latent_cycle": float(latent_cycle.detach()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    model.eval()
    return history, before, endpoint_metrics(model, images)


def update_manifest(args: argparse.Namespace) -> dict[str, object]:
    existing = load_exclusion_manifest(args.exclusion_manifest)
    entries = list(existing)
    for role, domain, path in (
        ("source", "cat", args.cat_path),
        ("target", "dog", args.dog_path),
    ):
        digest = sha256_file(path)
        if not any(entry["sha256"] == digest for entry in entries):
            entries.append(
                {
                    "role": role,
                    "domain": domain,
                    "original_split": "train",
                    "name": path.name,
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "reason": "deliberate single-pair overfit; exclude from all future train/test/eval",
                }
            )
    payload: dict[str, object] = {
        "version": 1,
        "policy": "Exclude every listed content hash from all future training, testing, and evaluation.",
        "entries": entries,
    }
    args.exclusion_manifest.write_text(json.dumps(payload, indent=2))
    (args.output_dir / "exclusion_manifest.json").write_text(json.dumps(payload, indent=2))
    return payload


@torch.no_grad()
def render_and_evaluate(
    model: LatentImageAutoencoder,
    cat_pil: Image.Image,
    dog_pil: Image.Image,
    images: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    endpoints = model.encode(images)
    dense_time = torch.linspace(0.0, 1.0, args.microsteps + 1, device=device)
    latent = torch.lerp(
        endpoints[0:1].expand(len(dense_time), -1, -1, -1),
        endpoints[1:2].expand(len(dense_time), -1, -1, -1),
        dense_time[:, None, None, None],
    )
    dense = model.decode(latent)
    display_index = torch.linspace(0, args.microsteps, DISPLAY_STEPS).round().long()
    display = dense[display_index]
    path_pil = [
        tensor_to_pil(frame.mul(2).sub(1)).resize((256, 256), Image.Resampling.LANCZOS)
        for frame in display.cpu()
    ]
    make_contact_sheet(
        [cat_pil],
        [path_pil],
        [dog_pil],
        args.output_dir / "single_pair_overfit.png",
        clock_label="memorized latent",
    )
    make_distribution_gif(
        [path_pil], args.output_dir / "single_pair_overfit.gif", clock_label="memorized latent"
    )
    dino = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb", image_size=64, pretrained=False, fusion="gated"
    )
    dino.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True))
    dino.eval().requires_grad_(False).to(device)
    preprocess = BatchPreprocessor(64)
    features, _, probability = dino_encode(dino, dense.cpu(), preprocess, device)
    endpoint_features = features[[0, -1]]
    direction = species_direction(endpoint_features[0:1], endpoint_features[1:2])
    coordinate = features @ direction
    normalized_progress = ((coordinate - coordinate[0]) / (coordinate[-1] - coordinate[0])).cpu()
    torch.save(
        {"dense_time": dense_time.cpu(), "dense_rgb": dense.cpu(), "dense_latent": latent.cpu()},
        args.output_dir / "single_pair_trajectory.pt",
    )
    return {
        "dense_dog_probability": probability.tolist(),
        "display_dog_probability": probability[display_index].tolist(),
        "dense_species_progress": normalized_progress.tolist(),
        "monotone_progress_fraction": float(
            (normalized_progress[1:] >= normalized_progress[:-1] - 1e-4).float().mean()
        ),
        "dense_adjacent_l1": float((dense[1:] - dense[:-1]).abs().mean()),
    }


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    afhq = project / "data/stargan-v2/data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat-path", type=Path, default=afhq / "train/cat/flickr_cat_000002.png")
    parser.add_argument("--dog-path", type=Path, default=afhq / "train/dog/flickr_dog_000002.png")
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=project / "outputs/latent_schrodinger_bridge/latent_autoencoder.pt",
    )
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt"
    )
    parser.add_argument("--exclusion-manifest", type=Path, default=afhq / "exclusions.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lambda-interior-anchor", type=float, default=0.01)
    parser.add_argument("--lambda-acceleration", type=float, default=0.005)
    parser.add_argument("--lambda-latent-cycle", type=float, default=0.01)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--microsteps", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("mps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cat_pil, cat = load_tensor(args.cat_path)
    dog_pil, dog = load_tensor(args.dog_path)
    images = torch.stack([cat, dog]).to(device)
    model = load_autoencoder(args, device)
    history, before, after = train(model, images, args)
    rendering = render_and_evaluate(model, cat_pil, dog_pil, images, args, device)
    manifest = update_manifest(args)
    # Constructing the datasets after writing the manifest is the executable
    # verification that neither memorized image can re-enter future splits.
    counts = {
        f"{split}_{domain}": len(
            AFHQDomainDataset(
                args.cat_path.parents[2],
                domain,
                image_size=128,
                training=split == "train",
                exclusion_manifest=args.exclusion_manifest,
            )
        )
        for split in ("train", "test")
        for domain in ("cat", "dog")
    }
    summary = {
        "status": "quarantined_memorization_sanity_check",
        "cat_path": str(args.cat_path),
        "dog_path": str(args.dog_path),
        "endpoint_before": before,
        "endpoint_after": after,
        "render": rendering,
        "post_exclusion_counts": counts,
        "excluded_hashes": [entry["sha256"] for entry in manifest["entries"]],
        "history": history,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(
        {"model": model.cpu().state_dict(), "summary": summary},
        args.output_dir / "single_pair_overfit_autoencoder.pt",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
