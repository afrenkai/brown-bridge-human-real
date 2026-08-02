"""Source-only cat-to-dog transport in the StarGAN-v2 AFHQ style manifold.

The earlier sharp demos used a paired dog style at inference.  This experiment
keeps their successful observation model but replaces the paired endpoint by a
LightSB conditional learned from independent cat and dog style marginals.  A
held-out path therefore needs only its source cat image.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brown_bridge.data import BatchPreprocessor
from brown_bridge.flow_matching import TimeConditionedMLP, integrate_flow
from brown_bridge.light_sb import (
    CoordinateStandardizer,
    DiagonalLightSB,
    brownian_bridge_path,
    expected_minibatch_ot_pairs,
)
from brown_bridge.models import DinoCatDogClassifier
from brown_bridge.probes import BinaryLinearProbe
from brown_bridge.stargan import Generator, StyleEncoder, load_afhq_renderer
from render_dino_stargan_bridge import (
    CAT_DOMAIN,
    DOG_DOMAIN,
    choose_clock_indices,
    dino_encode,
    make_distribution_gif,
    tensor_to_pil,
)
from render_full_dataset_demo import fit_or_load_afhq_probe
from train_continuous_vae_renderer import AFHQDomainDataset
from train_distributional_latent_transport import (
    endpoint_distribution_metrics,
    train_lightsb,
    train_rectified_flow,
)


@torch.inference_mode()
def encode_styles(
    encoder: StyleEncoder,
    dataset: AFHQDomainDataset,
    domain: int,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    result: list[Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_index, images in enumerate(loader, start=1):
        images = images.to(device).mul(2.0).sub(1.0)
        labels = torch.full((len(images),), domain, device=device, dtype=torch.long)
        result.append(encoder(images, labels).float().cpu())
        if batch_index % 25 == 0 or batch_index == len(loader):
            print(
                f"encoded domain={domain}: {min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    return torch.cat(result)


def style_cache(
    encoder: StyleEncoder,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, AFHQDomainDataset]]:
    datasets = {
        domain: AFHQDomainDataset(
            args.afhq_root,
            domain,
            image_size=256,
            training=True,
            augment=False,
            limit=args.train_limit,
        )
        for domain in ("cat", "dog")
    }
    names = {domain: [path.name for path in data.paths] for domain, data in datasets.items()}
    if args.style_cache.exists():
        payload = torch.load(args.style_cache, map_location="cpu", weights_only=False)
        if payload.get("names") != names:
            raise RuntimeError("style cache does not match the filtered AFHQ training data")
        return payload["cat"].float(), payload["dog"].float(), datasets
    styles = {
        "cat": encode_styles(
            encoder, datasets["cat"], CAT_DOMAIN,
            batch_size=args.encode_batch_size, device=device,
        ),
        "dog": encode_styles(
            encoder, datasets["dog"], DOG_DOMAIN,
            batch_size=args.encode_batch_size, device=device,
        ),
    }
    args.style_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**styles, "names": names}, args.style_cache)
    return styles["cat"], styles["dog"], datasets


@torch.inference_mode()
def render_style_path(
    renderer: Generator,
    sources: Tensor,
    standardized_path: Tensor,
    standardizer: CoordinateStandardizer,
    *,
    batch_size: int,
    preserve_observed_start: bool = True,
) -> Tensor:
    """Render ``[time,batch,style]`` as ``[batch,time,3,256,256]`` in [0,1]."""
    time_count, source_count, dimension = standardized_path.shape
    raw_style = standardizer.inverse(standardized_path).reshape(-1, dimension)
    repeated_source = sources[None].expand(time_count, -1, -1, -1, -1).reshape(
        -1, *sources.shape[1:]
    )
    rendered = [
        renderer(image, style).add(1.0).div(2.0).clamp(0.0, 1.0).cpu()
        for image, style in zip(repeated_source.split(batch_size), raw_style.split(batch_size))
    ]
    path = torch.cat(rendered).reshape(time_count, source_count, 3, 256, 256).transpose(0, 1)
    if preserve_observed_start:
        path[:, 0] = sources.cpu().add(1.0).div(2.0)
    return path


@torch.inference_mode()
def dino_path_features(
    dino: DinoCatDogClassifier,
    path: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    preprocess = BatchPreprocessor(64)
    batches: list[Tensor] = []
    for images in path.flatten(0, 1).split(64):
        features, _, _ = dino_encode(dino, images, preprocess, device)
        batches.append(features)
    return torch.cat(batches).reshape(path.shape[0], path.shape[1], -1)


def path_metrics(path: Tensor, features: Tensor, probe: BinaryLinearProbe) -> dict[str, object]:
    probability = probe.probabilities(features.flatten(0, 1)).reshape(path.shape[:2])
    delta = probability[:, 1:] - probability[:, :-1]
    speed = torch.linalg.vector_norm(features[:, 1:] - features[:, :-1], dim=2)
    return {
        "source_dog_probability_mean": float(probability[:, 0].mean()),
        "endpoint_dog_probability_mean": float(probability[:, -1].mean()),
        "endpoint_dog_fraction": float((probability[:, -1] >= 0.5).float().mean()),
        "semantic_monotone_step_fraction": float((delta >= -0.01).float().mean()),
        "semantic_monotone_path_fraction": float((delta >= -0.01).all(1).float().mean()),
        "mean_dino_feature_step": float(speed.mean()),
        "mean_adjacent_frame_l1": float((path[:, 1:] - path[:, :-1]).abs().mean()),
        "path_dog_probability": probability.tolist(),
    }


def semantic_subsample(path: Tensor, features: Tensor, direction: Tensor, count: int) -> tuple[Tensor, list[list[int]]]:
    selected: list[Tensor] = []
    indices: list[list[int]] = []
    coordinate = features @ F.normalize(direction.float(), dim=0)
    for row in range(len(path)):
        chosen, _ = choose_clock_indices(coordinate[row], count)
        indices.append(chosen)
        selected.append(path[row, chosen])
    return torch.stack(selected), indices


def pil_paths(path: Tensor) -> list[list[Image.Image]]:
    return [
        [tensor_to_pil(frame.mul(2.0).sub(1.0)) for frame in row]
        for row in path
    ]


def save_source_only_sheet(
    paths: dict[str, Tensor],
    output: Path,
    *,
    time_label: str,
) -> None:
    names = list(paths)
    examples, steps = paths[names[0]].shape[:2]
    tile, header, method_header = 256, 36, 32
    width = steps * tile
    height = header + len(names) * (method_header + examples * tile)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column in range(steps):
        label = "observed cat" if column == 0 else f"{time_label}={column/(steps-1):.2f}"
        draw.text((column * tile + 8, 12), label, fill="black")
    top = header
    for name in names:
        draw.text((8, top + 9), name, fill="black")
        top += method_header
        for row, images in enumerate(pil_paths(paths[name])):
            for column, image in enumerate(images):
                canvas.paste(image, (column * tile, top + row * tile))
        top += examples * tile
    canvas.save(output)


def save_probability_plot(metrics: dict[str, dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for name, result in metrics.items():
        probability = torch.tensor(result["path_dog_probability"])
        time = torch.linspace(0.0, 1.0, probability.shape[1])
        axis.plot(time, probability.mean(0), label=name)
        axis.fill_between(
            time, probability.quantile(0.1, dim=0), probability.quantile(0.9, dim=0), alpha=0.15
        )
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.set(xlabel="physical bridge time", ylabel="AFHQ DINO probe P(dog)", ylim=(-0.02, 1.02))
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def select_demo_rows(result: dict[str, object], count: int) -> list[int]:
    """Select semantically clean paths while leaving population metrics untouched."""
    probability = torch.tensor(result["path_dog_probability"])
    if count > len(probability):
        raise ValueError("demo count exceeds the evaluated population")
    delta = probability[:, 1:] - probability[:, :-1]
    backward_variation = torch.relu(-delta - 0.01).sum(1)
    score = probability[:, -1] - 2.0 * backward_variation - 0.5 * probability[:, 0]
    return score.argsort(descending=True)[:count].tolist()


def load_dino(args: argparse.Namespace, device: torch.device) -> DinoCatDogClassifier:
    model = DinoCatDogClassifier(
        "vit_tiny_patch16_dinov3_qkvb", image_size=64, pretrained=False, fusion="gated"
    )
    model.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(device)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("this image experiment requires Apple MPS")
    device = torch.device("mps")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    renderer, style_encoder = load_afhq_renderer(str(args.stargan_checkpoint), device=device)
    dino = load_dino(args, device)
    probe, species_direction, probe_test_metrics = fit_or_load_afhq_probe(dino, args, device)
    raw_cat, raw_dog, train_data = style_cache(style_encoder, args, device)
    standardizer = CoordinateStandardizer.fit(torch.cat([raw_cat, raw_dog]))
    cat = standardizer.transform(raw_cat)
    dog = standardizer.transform(raw_dog)

    checkpoint_path = args.output_dir / "style_transport.pt"
    if args.evaluate_only:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        standardizer = CoordinateStandardizer.from_state_dict(saved["standardizer"])
        lightsb = DiagonalLightSB(
            cat.shape[1], components=args.sb_components, epsilon=args.sb_epsilon,
            initial_scale=args.sb_initial_scale,
        )
        lightsb.load_state_dict(saved["lightsb"])
        lightsb = lightsb.eval().to(device)
        flow = TimeConditionedMLP(cat.shape[1], width=args.flow_width, depth=args.flow_depth)
        flow.load_state_dict(saved["flow"])
        flow = flow.eval().to(device)
        sb_history, flow_history = saved["sb_history"], saved["flow_history"]
    else:
        lightsb, sb_history = train_lightsb(cat, dog, args, device)
        paired_cat, paired_dog = expected_minibatch_ot_pairs(
            cat, dog,
            batch_size=args.ot_batch_size,
            batches=args.ot_batches,
            epsilon=args.ot_epsilon,
            sinkhorn_iterations=args.ot_iterations,
            generator=torch.Generator().manual_seed(args.seed + 3),
        )
        flow, flow_history = train_rectified_flow(paired_cat, paired_dog, args, device)
        torch.save(
            {
                "lightsb": lightsb.state_dict(),
                "flow": flow.state_dict(),
                "standardizer": standardizer.state_dict(),
                "sb_history": sb_history,
                "flow_history": flow_history,
            },
            checkpoint_path,
        )

    test = {
        domain: AFHQDomainDataset(
            args.afhq_root, domain, image_size=256, training=False, augment=False
        )
        for domain in ("cat", "dog")
    }
    rng = torch.Generator().manual_seed(args.seed + 5)
    cat_indices = torch.randperm(len(test["cat"]), generator=rng)[: args.eval_examples]
    dog_indices = torch.randperm(len(test["dog"]), generator=rng)[: args.eval_examples]
    cats = torch.stack([test["cat"][int(index)] for index in cat_indices])
    dogs = torch.stack([test["dog"][int(index)] for index in dog_indices])
    source_star = cats.to(device).mul(2.0).sub(1.0)
    cat_domain = torch.full((len(cats),), CAT_DOMAIN, device=device, dtype=torch.long)
    source_style = standardizer.transform(style_encoder(source_star, cat_domain))

    with torch.no_grad():
        mean_endpoint = lightsb.conditional_mean(source_style)
        sampled_endpoint = lightsb.sample_endpoint(
            source_style, generator=torch.Generator().manual_seed(args.seed + 6)
        )
        _, flow_dense = integrate_flow(
            flow,
            source_style,
            macro_steps=args.dense_steps - 1,
            microsteps_per_macro=args.flow_microsteps,
            method="heun",
            geometry="ambient",
        )
        time = torch.linspace(0.0, 1.0, args.dense_steps, device=device)
        mean_path = torch.lerp(source_style[None], mean_endpoint[None], time[:, None, None])
        sampled_path = brownian_bridge_path(
            source_style,
            sampled_endpoint,
            time,
            epsilon=args.sb_epsilon,
            generator=torch.Generator().manual_seed(args.seed + 7),
        )
        flow_path = flow_dense[:: args.flow_microsteps]

    standardized_paths = {
        "LightSB conditional mean": mean_path,
        "LightSB reciprocal SDE": sampled_path,
        "expected-minibatch-OT rectified flow": flow_path,
    }
    rendered: dict[str, Tensor] = {}
    for name, path in standardized_paths.items():
        rendered[name] = render_style_path(
            renderer, source_star, path, standardizer, batch_size=args.render_batch_size
        )
        print(f"rendered {name}", flush=True)

    preprocess = BatchPreprocessor(64)
    dog_features, _, _ = dino_encode(dino, dogs, preprocess, device)
    source_features, _, _ = dino_encode(dino, cats, preprocess, device)
    metrics: dict[str, dict[str, object]] = {}
    feature_paths: dict[str, Tensor] = {}
    for name, path in rendered.items():
        features = dino_path_features(dino, path, device=device)
        feature_paths[name] = features
        result = path_metrics(path, features, probe)
        result.update(
            endpoint_distribution_metrics(
                source_features, features[:, -1], dog_features, species_direction
            )
        )
        metrics[name] = result
        print(name + " " + json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}), flush=True)

    demo_rows = select_demo_rows(metrics["LightSB conditional mean"], args.examples)
    displayed: dict[str, Tensor] = {}
    semantic_indices: dict[str, list[list[int]]] = {}
    for name in rendered:
        selected, indices = semantic_subsample(
            rendered[name][demo_rows],
            feature_paths[name][demo_rows],
            species_direction,
            args.sheet_steps,
        )
        displayed[name] = selected
        semantic_indices[name] = indices
        make_distribution_gif(
            pil_paths(rendered[name][demo_rows]),
            args.output_dir / (name.lower().replace(" ", "_") + ".gif"),
            clock_label=name,
            duration_ms=args.frame_duration_ms,
        )
    save_source_only_sheet(
        displayed,
        args.output_dir / "source_only_style_transport.png",
        time_label="semantic t",
    )
    save_probability_plot(metrics, args.output_dir / "physical_time_dog_probability.png")

    summary: dict[str, object] = {
        "claim": "source-only cat-style to dog-style marginal transport",
        "inference": "Only a held-out cat image is encoded. No dog image or dog index is selected.",
        "sde": "Conditional on the LightSB terminal sample, dS_t = (S_1-S_t)/(1-t) dt + sqrt(epsilon) dW_t in standardized StarGAN style coordinates.",
        "observation": "X_t = G(X_0, standardizer^{-1}(S_t)); the StarGAN content stream is fixed to the observed cat.",
        "train_counts": {domain: len(data) for domain, data in train_data.items()},
        "test_counts": {domain: len(data) for domain, data in test.items()},
        "cat_test_indices": cat_indices.tolist(),
        "dog_evaluation_indices": dog_indices.tolist(),
        "demo_eval_rows": demo_rows,
        "demo_selection": "Highest LightSB mean endpoint P(dog), penalized for backward semantic variation and source P(dog); population metrics use all held-out paths.",
        "afhq_probe_test_metrics": probe_test_metrics,
        "metrics": metrics,
        "semantic_display_dense_indices": semantic_indices,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({name: path.cpu() for name, path in standardized_paths.items()}, args.output_dir / "style_paths.pt")
    return summary


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq-root", type=Path, default=project / "data/stargan-v2/data")
    parser.add_argument(
        "--stargan-checkpoint", type=Path,
        default=project / "data/stargan-v2/expr/checkpoints/afhq/100000_nets_ema.ckpt",
    )
    parser.add_argument("--dino-checkpoint", type=Path, default=project / "outputs/cat_dog/classifier.pt")
    parser.add_argument(
        "--probe-cache", type=Path,
        default=project / "outputs/full_dataset_latent_bridge_demo/afhq_dino_linear_probe.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/stargan_style_lightsb")
    parser.add_argument("--style-cache", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sb-components", type=int, default=32)
    parser.add_argument("--sb-epsilon", type=float, default=0.08)
    parser.add_argument("--sb-initial-scale", type=float, default=0.1)
    parser.add_argument("--sb-learning-rate", type=float, default=1e-3)
    parser.add_argument("--sb-steps", type=int, default=10000)
    parser.add_argument("--lambda-scale", type=float, default=1e-2)
    parser.add_argument("--lambda-balance", type=float, default=1.0)
    parser.add_argument("--ot-batch-size", type=int, default=128)
    parser.add_argument("--ot-batches", type=int, default=256)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iterations", type=int, default=40)
    parser.add_argument("--flow-width", type=int, default=256)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument("--flow-learning-rate", type=float, default=3e-4)
    parser.add_argument("--flow-steps", type=int, default=6000)
    parser.add_argument("--flow-microsteps", type=int, default=2)
    parser.add_argument("--dense-steps", type=int, default=25)
    parser.add_argument("--sheet-steps", type=int, default=7)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--frame-duration-ms", type=int, default=160)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.style_cache is None:
        args.style_cache = args.output_dir / "train_stargan_styles.pt"
    if not 2 <= args.sheet_steps <= args.dense_steps:
        parser.error("sheet steps must lie between two and dense steps")
    if not 1 <= args.examples <= args.eval_examples:
        parser.error("examples must be positive and no larger than eval examples")
    return args


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(run(configuration), indent=2))
