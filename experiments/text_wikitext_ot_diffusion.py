"""Bidirectional OT diffusion on topic slices of the WikiText-2 corpus."""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from brown_bridge.bridges import brownian_interpolant
from brown_bridge.exploration import best_of_k_loss
from brown_bridge.ot import sinkhorn
from brown_bridge.paths import sample_coupling
from brown_bridge.schemas import CoupledEndpoints


@dataclass(frozen=True)
class TextRecord:
    text: str
    title: str


def ensure_wikitext(root: Path, *, download: bool) -> Path:
    raw_path = root / "wikitext-2-raw/wiki.train.raw"
    if raw_path.exists():
        return raw_path
    if not download:
        raise FileNotFoundError(
            f"missing {raw_path}; rerun with --download to fetch WikiText-2"
        )
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "wikitext-2-raw-v1.zip"
    dataset_url = (
        "https://huggingface.co/datasets/ggml-org/ci/resolve/main/"
        "wikitext-2-raw-v1.zip"
    )
    urllib.request.urlretrieve(dataset_url, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    return raw_path


def load_wikitext_records(raw_path: Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    current_title = "unknown"
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        heading = re.match(r"^=+\s+(.+?)\s+=+$", stripped)
        if heading:
            current_title = heading.group(1)
            continue
        if not stripped or stripped.startswith("="):
            continue
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
        if 24 <= len(tokens) <= 180:
            records.append(TextRecord(" ".join(tokens), current_title))
    return records


def select_topic_records(
    records: list[TextRecord],
    keywords: list[str],
    *,
    count: int,
    generator: torch.Generator,
) -> list[TextRecord]:
    keyword_set = {keyword.lower() for keyword in keywords}
    scored_records = [
        record
        for record in records
        if keyword_set.intersection(
            re.findall(r"[A-Za-z][A-Za-z'-]*", f"{record.title} {record.text}".lower())
        )
    ]
    if len(scored_records) < count:
        raise ValueError(
            f"topic slice has {len(scored_records)} records but needs {count}; "
            f"use broader keywords or a smaller --sample-count"
        )
    permutation = torch.randperm(len(scored_records), generator=generator)[:count]
    return [scored_records[int(index)] for index in permutation]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]*", text)]


def tfidf_features(texts: list[str], *, vocabulary_size: int) -> Tensor:
    tokenized = [tokenize(text) for text in texts]
    document_frequency: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    ranked_tokens = sorted(
        document_frequency,
        key=lambda token: (-document_frequency[token], token),
    )[:vocabulary_size]
    vocabulary = {token: index for index, token in enumerate(ranked_tokens)}
    inverse_document_frequency = torch.tensor(
        [
            math.log((1.0 + len(texts)) / (1.0 + document_frequency[token])) + 1.0
            for token in ranked_tokens
        ],
        dtype=torch.float32,
    )
    features = torch.zeros((len(texts), len(vocabulary)), dtype=torch.float32)
    for row, tokens in enumerate(tokenized):
        counts: dict[str, int] = {}
        for token in tokens:
            if token in vocabulary:
                counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            features[row, vocabulary[token]] = count / max(len(tokens), 1)
    features.mul_(inverse_document_frequency)
    return F.normalize(features, dim=1)


def make_latents(
    source_texts: list[str],
    target_texts: list[str],
    *,
    vocabulary_size: int,
    latent_dimension: int,
) -> tuple[Tensor, Tensor]:
    all_texts = source_texts + target_texts
    features = tfidf_features(all_texts, vocabulary_size=vocabulary_size)
    centered = features - features.mean(0, keepdim=True)
    component_count = min(latent_dimension, centered.shape[0] - 1, centered.shape[1])
    if component_count < 2:
        raise ValueError("the selected corpus slice has too few TF-IDF dimensions")
    _, singular_values, components = torch.pca_lowrank(
        centered,
        q=component_count,
        center=False,
    )
    latent = centered @ components[:, :component_count]
    latent = latent / singular_values[:component_count].clamp_min(1e-4)
    return latent[: len(source_texts)], latent[len(source_texts) :]


class BridgeDiffusionNetwork(nn.Module):
    def __init__(
        self, latent_dimension: int, candidate_count: int, hidden_dimension: int
    ) -> None:
        super().__init__()
        self.latent_dimension = latent_dimension
        self.candidate_count = candidate_count
        self.network = nn.Sequential(
            nn.Linear(latent_dimension + 4, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 2 * candidate_count * latent_dimension),
        )

    def forward(
        self, state: Tensor, time: Tensor, direction: Tensor
    ) -> tuple[Tensor, Tensor]:
        features = torch.cat(
            [
                state,
                time[:, None],
                torch.sin(2.0 * torch.pi * time)[:, None],
                torch.cos(2.0 * torch.pi * time)[:, None],
                direction[:, None],
            ],
            dim=1,
        )
        prediction = self.network(features).reshape(
            len(state), self.candidate_count, 2, self.latent_dimension
        )
        return prediction[:, :, 0], prediction[:, :, 1]


def bridge_training_batch(
    endpoints: CoupledEndpoints,
    batch_size: int,
    *,
    epsilon: float,
    minimum_time: float,
    direction: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = torch.randint(
        len(endpoints.source),
        (batch_size,),
        generator=generator,
    )
    source = endpoints.source[indices]
    target = endpoints.target[indices]
    time = minimum_time + (1.0 - 2.0 * minimum_time) * torch.rand(
        batch_size,
        generator=generator,
    )
    noise = torch.randn(source.shape, generator=generator)
    state = brownian_interpolant(source, target, time, noise, epsilon=epsilon)
    variance = epsilon * time * (1.0 - time)
    score_target = -noise / variance.sqrt()[:, None]
    bridge_drift_target = (target - state) / (1.0 - time)[:, None]
    direction_tensor = torch.full_like(time, direction)
    return (
        state,
        time,
        direction_tensor,
        torch.stack([bridge_drift_target, score_target], dim=1),
    )


def train_bridge(
    endpoints: CoupledEndpoints,
    *,
    latent_dimension: int,
    candidate_count: int,
    hidden_dimension: int,
    batch_size: int,
    steps: int,
    epsilon: float,
    minimum_time: float,
    score_weight: float,
    learning_rate: float,
    seed: int,
) -> tuple[BridgeDiffusionNetwork, list[float]]:
    model = BridgeDiffusionNetwork(latent_dimension, candidate_count, hidden_dimension)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for step_index in range(steps):
        forward = bridge_training_batch(
            endpoints,
            batch_size,
            epsilon=epsilon,
            minimum_time=minimum_time,
            direction=1.0,
            generator=generator,
        )
        reverse_endpoints = CoupledEndpoints(
            endpoints.target,
            endpoints.source,
            endpoints.target_indices,
            endpoints.source_indices,
        )
        reverse = bridge_training_batch(
            reverse_endpoints,
            batch_size,
            epsilon=epsilon,
            minimum_time=minimum_time,
            direction=-1.0,
            generator=generator,
        )
        state = torch.cat([forward[0], reverse[0]])
        time = torch.cat([forward[1], reverse[1]])
        direction = torch.cat([forward[2], reverse[2]])
        targets = torch.cat([forward[3], reverse[3]])
        drift_prediction, score_prediction = model(state, time, direction)
        candidate_losses = (drift_prediction - targets[:, 0, None, :]).square().mean(
            -1
        ) + score_weight * (score_prediction - targets[:, 1, None, :]).square().mean(-1)
        result = best_of_k_loss(candidate_losses)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        optimizer.step()
        losses.append(float(result.loss.detach()))
    return model, losses


@torch.no_grad()
def rollout_bridge(
    model: BridgeDiffusionNetwork,
    initial: Tensor,
    destination: Tensor,
    *,
    direction: float,
    epsilon: float,
    steps: int,
    minimum_time: float,
    generator: torch.Generator,
) -> Tensor:
    state = initial.clone()
    path = [state.clone()]
    step_size = 1.0 / steps
    for step in range(steps):
        time_value = min(
            max((step + 0.5) * step_size, minimum_time),
            1.0 - minimum_time,
        )
        time = torch.full((len(state),), time_value)
        direction_tensor = torch.full_like(time, direction)
        drift_candidates, _ = model(state, time, direction_tensor)
        candidate_state = state[:, None] + step_size * drift_candidates
        candidate_state = candidate_state + math.sqrt(
            epsilon * step_size
        ) * torch.randn(
            candidate_state.shape,
            generator=generator,
        )
        target_distance = torch.cdist(
            candidate_state.flatten(0, 1), destination
        ).square()
        target_distance = target_distance.min(dim=1).values.reshape(len(state), -1)
        winner = target_distance.argmin(dim=1)
        state = candidate_state[torch.arange(len(state)), winner]
        path.append(state.clone())
    return torch.stack(path)


def nearest_text(
    latent: Tensor, reference_latents: Tensor, reference_texts: list[str]
) -> list[str]:
    nearest = torch.cdist(latent, reference_latents).square().argmin(dim=1)
    return [reference_texts[int(index)] for index in nearest]


def plot_results(
    source_latents: Tensor,
    target_latents: Tensor,
    forward_path: Tensor,
    reverse_path: Tensor,
    output_path: Path,
) -> None:
    all_latents = torch.cat([source_latents, target_latents])
    centered = all_latents - all_latents.mean(0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=3, center=False)
    source_coordinates = (source_latents - all_latents.mean(0)) @ components[:, :3]
    target_coordinates = (target_latents - all_latents.mean(0)) @ components[:, :3]
    forward_coordinates = torch.einsum(
        "tbd,dk->tbk",
        forward_path - all_latents.mean(0),
        components[:, :3],
    )
    reverse_coordinates = torch.einsum(
        "tbd,dk->tbk",
        reverse_path - all_latents.mean(0),
        components[:, :3],
    )
    source_count = len(source_latents)
    figure = plt.figure(figsize=(12, 5), constrained_layout=True)
    distribution_axis = figure.add_subplot(1, 2, 1, projection="3d")
    trajectory_axis = figure.add_subplot(1, 2, 2, projection="3d")
    distribution_axis.scatter(
        source_coordinates[:, 0],
        source_coordinates[:, 1],
        source_coordinates[:, 2],
        s=12,
        label="source A",
    )
    distribution_axis.scatter(
        target_coordinates[:, 0],
        target_coordinates[:, 1],
        target_coordinates[:, 2],
        s=12,
        label="target B",
    )
    distribution_axis.scatter(
        forward_coordinates[-1, :, 0],
        forward_coordinates[-1, :, 1],
        forward_coordinates[-1, :, 2],
        s=12,
        label="A → B",
    )
    distribution_axis.scatter(
        reverse_coordinates[-1, :, 0],
        reverse_coordinates[-1, :, 1],
        reverse_coordinates[-1, :, 2],
        s=12,
        label="B → A",
    )
    distribution_axis.set_xlabel("PC1")
    distribution_axis.set_ylabel("PC2")
    distribution_axis.set_zlabel("PC3")
    distribution_axis.set_title("WikiText-2 latent distributions")
    distribution_axis.legend(fontsize=8)
    for index in range(min(16, source_count)):
        trajectory_axis.plot(
            forward_coordinates[:, index, 0],
            forward_coordinates[:, index, 1],
            forward_coordinates[:, index, 2],
            color="tab:blue",
            alpha=0.35,
        )
        trajectory_axis.plot(
            reverse_coordinates[:, index, 0],
            reverse_coordinates[:, index, 1],
            reverse_coordinates[:, index, 2],
            color="tab:orange",
            alpha=0.35,
        )
    trajectory_axis.scatter(
        source_coordinates[:, 0],
        source_coordinates[:, 1],
        source_coordinates[:, 2],
        s=8,
        color="tab:blue",
        label="A",
    )
    trajectory_axis.scatter(
        target_coordinates[:, 0],
        target_coordinates[:, 1],
        target_coordinates[:, 2],
        s=8,
        color="tab:orange",
        label="B",
    )
    trajectory_axis.set_xlabel("PC1")
    trajectory_axis.set_ylabel("PC2")
    trajectory_axis.set_zlabel("PC3")
    trajectory_axis.set_title("256-step bridge trajectories")
    trajectory_axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/wikitext-2"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--source-keywords",
        default="computer,software,technology,science,research,mathematics,physics",
    )
    parser.add_argument(
        "--target-keywords", default="music,film,television,actor,football,album,novel"
    )
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--vocabulary-size", type=int, default=1_024)
    parser.add_argument("--latent-dimension", type=int, default=16)
    parser.add_argument("--exploration-candidates", type=int, default=4)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--ot-epsilon", type=float, default=0.15)
    parser.add_argument("--bridge-epsilon", type=float, default=0.01)
    parser.add_argument("--ot-iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/text_wikitext_ot_diffusion")
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.sample_count < 4 or args.exploration_candidates < 1:
        raise ValueError(
            "sample-count must be at least four and candidates must be positive"
        )
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    raw_path = ensure_wikitext(args.data_root, download=args.download)
    records = load_wikitext_records(raw_path)
    source_records = select_topic_records(
        records,
        args.source_keywords.split(","),
        count=args.sample_count,
        generator=generator,
    )
    target_records = select_topic_records(
        records,
        args.target_keywords.split(","),
        count=args.sample_count,
        generator=generator,
    )
    source_texts = [record.text for record in source_records]
    target_texts = [record.text for record in target_records]
    source_latents, target_latents = make_latents(
        source_texts,
        target_texts,
        vocabulary_size=args.vocabulary_size,
        latent_dimension=args.latent_dimension,
    )
    ot_result = sinkhorn(
        source_latents,
        target_latents,
        epsilon=args.ot_epsilon,
        max_iterations=args.ot_iterations,
        cost="sqeuclidean",
    )
    endpoints = sample_coupling(
        ot_result.plan,
        source_latents,
        target_latents,
        max(args.sample_count * 8, 512),
        generator=generator,
    )
    model, losses = train_bridge(
        endpoints,
        latent_dimension=source_latents.shape[1],
        candidate_count=args.exploration_candidates,
        hidden_dimension=args.hidden_dimension,
        batch_size=args.batch_size,
        steps=args.steps,
        epsilon=args.bridge_epsilon,
        minimum_time=0.03,
        score_weight=0.1,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
    )
    forward_path = rollout_bridge(
        model,
        source_latents,
        target_latents,
        direction=1.0,
        epsilon=args.bridge_epsilon,
        steps=args.rollout_steps,
        minimum_time=0.03,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )
    reverse_path = rollout_bridge(
        model,
        target_latents,
        source_latents,
        direction=-1.0,
        epsilon=args.bridge_epsilon,
        steps=args.rollout_steps,
        minimum_time=0.03,
        generator=torch.Generator().manual_seed(args.seed + 3),
    )
    forward_nearest = nearest_text(forward_path[-1], target_latents, target_texts)
    reverse_nearest = nearest_text(reverse_path[-1], source_latents, source_texts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        source_latents,
        target_latents,
        forward_path,
        reverse_path,
        args.output_dir / "wikitext_ot_diffusion.png",
    )
    examples = []
    for index in range(min(8, args.sample_count)):
        examples.append(
            {
                "source": source_texts[index],
                "forward_nearest_target": forward_nearest[index],
                "target_start_for_reverse": target_texts[index],
                "reverse_nearest_source": reverse_nearest[index],
            }
        )
    (args.output_dir / "examples.json").write_text(
        json.dumps(examples, indent=2) + "\n"
    )
    summary = {
        "dataset": "WikiText-2 train.raw",
        "method": "entropic OT plus bidirectional Brownian-bridge score/drift diffusion",
        "flow_matching": False,
        "rectified_flow": False,
        "source_count": len(source_texts),
        "target_count": len(target_texts),
        "source_keywords": args.source_keywords.split(","),
        "target_keywords": args.target_keywords.split(","),
        "latent_dimension": int(source_latents.shape[1]),
        "ot_transport_cost": float(ot_result.transport_cost),
        "exploration_candidates": args.exploration_candidates,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "forward_terminal_target_distance": float(
            torch.cdist(forward_path[-1], target_latents).square().min(1).values.mean()
        ),
        "reverse_terminal_source_distance": float(
            torch.cdist(reverse_path[-1], source_latents).square().min(1).values.mean()
        ),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
