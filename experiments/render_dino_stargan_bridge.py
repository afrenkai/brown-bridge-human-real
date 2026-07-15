"""Render an interpretable cat→dog distribution path at native AFHQ resolution.

DINO supplies a content-aware source/reference coupling and a scalar clock for
species progression.  StarGAN-v2 is only the image observation model: its
64-dimensional style interpolation is densely sampled, and DINO chooses the
frames closest to equally spaced class-progress levels.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

from brown_bridge.data import BatchPreprocessor
from brown_bridge.features import (
    content_cost_matrix,
    content_descriptor,
    species_direction,
)
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.stargan import load_afhq_renderer


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_STARGAN = WORKSPACE / "data/stargan-v2"
CAT_DOMAIN = 0
DOG_DOMAIN = 1
DENSE_STEPS = 25
DISPLAY_STEPS = 7
SEED = 73


def require_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires an available MPS device")
    return torch.device("mps")


def load_square(path: Path, size: int = 256) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), (size, size)).copy()


def image_batch(images: list[Image.Image]) -> Tensor:
    return torch.stack([pil_to_tensor(image).float() / 255.0 for image in images])


@torch.inference_mode()
def dino_encode(
    model: DinoCatDogClassifier,
    images: Tensor,
    preprocess: BatchPreprocessor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    normalized = preprocess(images.to(device), training=False)
    features, _, patches = model.encode_spatial(normalized)
    if patches is None:
        raise RuntimeError("DINO backbone did not return spatial tokens")
    logits = model.classifier(features)
    return (
        F.normalize(features.float(), dim=1).cpu(),
        F.normalize(patches.float(), dim=2).cpu(),
        logits.softmax(1)[:, DOG_DOMAIN].float().cpu(),
    )


def unique_minimum_assignment(cost: Tensor) -> list[int]:
    """Exact rectangular assignment for the five-by-eight demo set."""
    rows, columns = cost.shape
    if rows > columns:
        raise ValueError("assignment requires at least as many targets as sources")
    row_index = torch.arange(rows)
    best_value = float("inf")
    best: tuple[int, ...] | None = None
    for candidate in itertools.permutations(range(columns), rows):
        value = float(cost[row_index, torch.tensor(candidate)].sum())
        if value < best_value:
            best_value = value
            best = candidate
    if best is None:
        raise RuntimeError("empty assignment")
    return list(best)


def choose_clock_indices(scores: Tensor, count: int) -> tuple[list[int], Tensor]:
    """Use a monotone DINO class coordinate to reparameterize generator time."""
    if scores.ndim != 1 or len(scores) < count:
        raise ValueError("need a dense one-dimensional score path")
    denominator = scores[-1] - scores[0]
    if abs(float(denominator)) < 1e-6:
        progress = torch.linspace(0.0, 1.0, len(scores))
    else:
        progress = ((scores - scores[0]) / denominator).clamp(0.0, 1.0)
        progress = torch.cummax(progress, dim=0).values
        progress[-1] = 1.0
    targets = torch.linspace(0.0, 1.0, count)
    # Exactly optimize the few interior display locations, keeping endpoints.
    best_loss = float("inf")
    best = tuple(range(count))
    for interior in itertools.combinations(range(1, len(scores) - 1), count - 2):
        candidate = (0, *interior, len(scores) - 1)
        loss = float((progress[list(candidate)] - targets).abs().sum())
        if loss < best_loss:
            best_loss = loss
            best = candidate
    return list(best), progress


def tensor_to_pil(image: Tensor) -> Image.Image:
    array = (
        image.detach()
        .float()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def make_contact_sheet(
    sources: list[Image.Image],
    paths: list[list[Image.Image]],
    references: list[Image.Image],
    output: Path,
    *,
    clock_label: str,
) -> None:
    if not paths or any(len(path) != len(paths[0]) for path in paths):
        raise ValueError("paths must be nonempty and have a common time grid")
    display_steps = len(paths[0])
    labels = ["cat: t=0"] + [
        f"{clock_label} t={i/(display_steps-1):.2f}"
        for i in range(1, display_steps)
    ]
    labels.append("paired dog ref")
    tile = 256
    header = 38
    canvas = Image.new("RGB", (len(labels) * tile, header + len(paths) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * tile + 8, 12), label, fill="black")
    for row, (source, path, reference) in enumerate(zip(sources, paths, references)):
        images = [source, *path[1:], reference]
        for column, image in enumerate(images):
            canvas.paste(image, (column * tile, header + row * tile))
    canvas.save(output)


def make_distribution_gif(
    paths: list[list[Image.Image]],
    output: Path,
    *,
    clock_label: str,
    duration_ms: int = 750,
) -> None:
    if not paths or any(len(path) != len(paths[0]) for path in paths):
        raise ValueError("paths must be nonempty and have a common time grid")
    display_steps = len(paths[0])
    tile = 192
    header = 34
    frames: list[Image.Image] = []
    for step in range(display_steps):
        canvas = Image.new("RGB", (len(paths) * tile, header + tile), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 11),
            f"Cat → dog held-out paths, {clock_label} time {step/(display_steps-1):.2f}",
            fill="black",
        )
        for column, path in enumerate(paths):
            canvas.paste(path[step].resize((tile, tile), Image.Resampling.LANCZOS), (column * tile, header))
        frames.append(canvas)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(SEED)
    device = require_mps()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    source_paths = sorted((args.assets / "src/cat").glob("*.jpg"))
    reference_paths = sorted((args.assets / "ref/dog").glob("*.jpg"))
    if len(source_paths) < 5 or len(reference_paths) < 5:
        raise FileNotFoundError("expected representative AFHQ cat sources and dog references")
    sources = [load_square(path) for path in source_paths]
    dog_pool = [load_square(path) for path in reference_paths]
    source_tensor = image_batch(sources)
    dog_tensor = image_batch(dog_pool)

    dino = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb",
        image_size=64,
        pretrained=False,
        fusion="gated",
    )
    dino.load_state_dict(
        torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True)
    )
    dino.eval().requires_grad_(False).to(device)
    preprocess = BatchPreprocessor(64)
    cat_global, cat_patch, cat_probability = dino_encode(
        dino, source_tensor, preprocess, device
    )
    dog_global, dog_patch, dog_probability = dino_encode(
        dino, dog_tensor, preprocess, device
    )
    direction = species_direction(cat_global, dog_global)
    cat_descriptor = content_descriptor(cat_global, cat_patch, direction)
    dog_descriptor = content_descriptor(dog_global, dog_patch, direction)
    content_cost = content_cost_matrix(cat_descriptor, dog_descriptor)
    assignment = unique_minimum_assignment(content_cost)
    paired_dogs = [dog_pool[index] for index in assignment]
    paired_dog_tensor = dog_tensor[assignment]

    generator, style_encoder = load_afhq_renderer(
        str(args.stargan_checkpoint), device=device
    )
    source_star = (2.0 * source_tensor - 1.0).to(device)
    dog_star = (2.0 * paired_dog_tensor - 1.0).to(device)
    cat_domains = torch.full((len(sources),), CAT_DOMAIN, device=device, dtype=torch.long)
    dog_domains = torch.full((len(sources),), DOG_DOMAIN, device=device, dtype=torch.long)
    cat_style = style_encoder(source_star, cat_domains)
    dog_style = style_encoder(dog_star, dog_domains)

    dense_times = torch.linspace(0.0, 1.0, DENSE_STEPS, device=device)
    dense_paths: list[Tensor] = []
    selected_indices: list[list[int]] = []
    clock_progress: list[list[float]] = []
    rendered_paths: list[list[Image.Image]] = []
    linear_paths: list[list[Image.Image]] = []
    path_probabilities: list[list[float]] = []
    path_coordinates: list[list[float]] = []
    style_times: list[list[float]] = []
    for row in range(len(sources)):
        styles = torch.lerp(
            cat_style[row].expand(DENSE_STEPS, -1),
            dog_style[row].expand(DENSE_STEPS, -1),
            dense_times[:, None],
        )
        rendered_batches: list[Tensor] = []
        for left in range(0, DENSE_STEPS, args.render_batch_size):
            right = min(left + args.render_batch_size, DENSE_STEPS)
            rendered_batches.append(
                generator(source_star[row : row + 1].expand(right - left, -1, -1, -1), styles[left:right]).cpu()
            )
        dense = torch.cat(rendered_batches)
        dense_paths.append(dense)
        dense_01 = dense.add(1.0).div(2.0).clamp(0.0, 1.0)
        dense_global, _, dense_probability = dino_encode(
            dino, dense_01, preprocess, device
        )
        coordinate = dense_global @ direction
        indices, progress = choose_clock_indices(coordinate, DISPLAY_STEPS)
        selected_indices.append(indices)
        clock_progress.append([float(value) for value in progress[indices]])
        chosen = [tensor_to_pil(dense[index]) for index in indices]
        # Make t=0 the actual observed cat, while subsequent states are generated.
        chosen[0] = sources[row]
        rendered_paths.append(chosen)
        linear_indices = torch.linspace(
            0, DENSE_STEPS - 1, DISPLAY_STEPS
        ).round().long().tolist()
        linear_chosen = [tensor_to_pil(dense[index]) for index in linear_indices]
        linear_chosen[0] = sources[row]
        linear_paths.append(linear_chosen)
        selected_tensor = torch.stack(
            [source_tensor[row], *[dense_01[index] for index in indices[1:]]]
        )
        selected_global, _, selected_probability = dino_encode(
            dino, selected_tensor, preprocess, device
        )
        path_probabilities.append([float(value) for value in selected_probability])
        path_coordinates.append([float(value) for value in selected_global @ direction])
        style_times.append([0.0, *[float(dense_times[index]) for index in indices[1:]]])

    make_contact_sheet(
        sources,
        rendered_paths,
        paired_dogs,
        output / "cat_to_dog_contact_sheet.png",
        clock_label="DINO",
    )
    make_distribution_gif(
        rendered_paths,
        output / "cat_to_dog_distribution.gif",
        clock_label="DINO",
    )
    make_contact_sheet(
        sources,
        linear_paths,
        paired_dogs,
        output / "cat_to_dog_linear_contact_sheet.png",
        clock_label="style",
    )
    make_distribution_gif(
        linear_paths,
        output / "cat_to_dog_linear_distribution.gif",
        clock_label="style",
    )

    endpoint_images = torch.stack(
        [dense_paths[row][selected_indices[row][-1]].add(1).div(2).clamp(0, 1) for row in range(len(sources))]
    )
    endpoint_global, _, endpoint_probability = dino_encode(
        dino, endpoint_images, preprocess, device
    )
    monotone_fraction = sum(
        all(b >= a - 1e-5 for a, b in zip(values, values[1:]))
        for values in path_coordinates
    ) / len(path_coordinates)
    summary: dict[str, object] = {
        "device": str(device),
        "sources": [path.name for path in source_paths],
        "dog_reference_pool": [path.name for path in reference_paths],
        "assignment": assignment,
        "paired_references": [reference_paths[index].name for index in assignment],
        "paired_content_cost": [float(content_cost[row, column]) for row, column in enumerate(assignment)],
        "source_dino_dog_probability": [float(value) for value in cat_probability],
        "reference_dino_dog_probability": [float(dog_probability[index]) for index in assignment],
        "endpoint_dino_dog_probability": [float(value) for value in endpoint_probability],
        "display_path_dog_probability": path_probabilities,
        "display_path_species_coordinate": path_coordinates,
        "selected_dense_indices": selected_indices,
        "selected_style_times": style_times,
        "selected_monotone_clock": clock_progress,
        "coordinate_monotone_path_fraction": monotone_fraction,
        "dino_endpoint_to_paired_reference_cosine": [
            float(F.cosine_similarity(endpoint_global[row], dog_global[column], dim=0))
            for row, column in enumerate(assignment)
        ],
        "renderer": "StarGAN-v2 AFHQ EMA, 100000 iterations",
        "transport_interpretation": "DINO content assignment plus DINO-reparameterized StarGAN style bridge",
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        type=Path,
        default=DEFAULT_STARGAN / "assets/representative/afhq",
    )
    parser.add_argument(
        "--stargan-checkpoint",
        type=Path,
        default=DEFAULT_STARGAN / "expr/checkpoints/afhq/100000_nets_ema.ckpt",
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default=WORKSPACE / "outputs/cat_dog/classifier.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/stargan_dino_bridge",
    )
    parser.add_argument("--render-batch-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
