"""Metal-feasible continuous text codec pretraining on OpenWebText.

The script trains a decoder-compatible continuous slot codec before any
sentiment or ontology transport.  Frozen T5 encoder states are compressed into
slots and expanded back into continuous encoder states that a calibrated T5
decoder can read.  The codec is trained with reconstruction, latent round-trip
consistency, and a masked-context joint-embedding predictive loss.

No token probabilities are transported, and no flow, diffusion, velocity
field, or ODE solver is trained here.  This is representation pretraining for
the static K=1 Monge atlas in ``text_sst2_t5_slot_monge_ot.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from einops import rearrange
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


@dataclass(frozen=True)
class TokenizedCorpus:
    """CPU-resident token boundaries for a deterministic document subset."""

    documents: list[str]
    input_ids: Tensor
    attention_mask: Tensor


def device_from_argument(device_name: str) -> torch.device:
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested, but Metal MPS is unavailable")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def normalize_document(document: str) -> str:
    return " ".join(document.split())


def stream_openwebtext_documents(arguments: argparse.Namespace) -> list[str]:
    """Read a fixed shuffled OpenWebText subset without materializing the corpus."""
    required_count = arguments.train_document_count + arguments.validation_document_count
    dataset = load_dataset(
        arguments.dataset_name,
        split=arguments.dataset_split,
        streaming=True,
        cache_dir=str(arguments.data_cache_root),
    )
    shard_index = arguments.source_shard
    if shard_index < 0:
        shard_index = arguments.seed % arguments.dataset_shard_count
    source_shard = dataset.shard(
        num_shards=arguments.dataset_shard_count,
        index=shard_index,
        contiguous=True,
    )
    shuffled = source_shard.shuffle(seed=arguments.seed, buffer_size=arguments.shuffle_buffer)
    documents: list[str] = []
    progress = tqdm(total=required_count, desc="Streaming OpenWebText", unit="document", dynamic_ncols=True)
    for record in shuffled:
        document = normalize_document(record[arguments.text_column])
        if len(document) < arguments.minimum_document_characters:
            continue
        documents.append(document)
        progress.update(1)
        if len(documents) == required_count:
            break
    progress.close()
    if len(documents) != required_count:
        raise RuntimeError(
            f"OpenWebText stream produced {len(documents)} usable documents, expected {required_count}"
        )
    return documents


def tokenize_documents(
    tokenizer: object,
    documents: list[str],
    *,
    maximum_length: int,
) -> TokenizedCorpus:
    tokenized = tokenizer(
        documents,
        padding="max_length",
        truncation=True,
        max_length=maximum_length,
        return_tensors="pt",
    )
    return TokenizedCorpus(
        documents=documents,
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
    )


def cache_metadata_path(output_dir: Path) -> Path:
    return output_dir / "frozen_t5_states_metadata.json"


def frozen_state_cache_matches(
    output_dir: Path,
    *,
    document_count: int,
    maximum_length: int,
    state_dimension: int,
    dataset_name: str,
    dataset_shard_count: int,
    source_shard: int,
    seed: int,
) -> bool:
    state_path = output_dir / "frozen_t5_states.npy"
    metadata_path = cache_metadata_path(output_dir)
    if not state_path.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata == {
        "document_count": document_count,
        "maximum_length": maximum_length,
        "state_dimension": state_dimension,
        "dataset_name": dataset_name,
        "dataset_shard_count": dataset_shard_count,
        "source_shard": source_shard,
        "seed": seed,
        "dtype": "float16",
    }


@torch.inference_mode()
def cache_frozen_encoder_states(
    model: T5ForConditionalGeneration,
    corpus: TokenizedCorpus,
    *,
    arguments: argparse.Namespace,
    device: torch.device,
) -> np.memmap:
    """Encode once, saving compact frozen states for repeated MPS codec epochs."""
    state_dimension = model.config.d_model
    state_path = arguments.output_dir / "frozen_t5_states.npy"
    source_shard = arguments.source_shard if arguments.source_shard >= 0 else arguments.seed % arguments.dataset_shard_count
    cache_matches = frozen_state_cache_matches(
        arguments.output_dir,
        document_count=len(corpus.documents),
        maximum_length=arguments.maximum_length,
        state_dimension=state_dimension,
        dataset_name=arguments.dataset_name,
        dataset_shard_count=arguments.dataset_shard_count,
        source_shard=source_shard,
        seed=arguments.seed,
    )
    if cache_matches:
        return np.load(state_path, mmap_mode="r")
    state_cache = np.lib.format.open_memmap(
        state_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(corpus.documents), arguments.maximum_length, state_dimension),
    )
    model.encoder.eval()
    for start in tqdm(
        range(0, len(corpus.documents), arguments.encoder_batch_size),
        desc="Caching frozen T5 states",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(start + arguments.encoder_batch_size, len(corpus.documents))
        states = model.encoder(
            input_ids=corpus.input_ids[start:stop].to(device),
            attention_mask=corpus.attention_mask[start:stop].to(device),
            return_dict=True,
        ).last_hidden_state
        state_cache[start:stop] = states.float().cpu().numpy().astype(np.float16)
    state_cache.flush()
    cache_metadata_path(arguments.output_dir).write_text(
        json.dumps(
            {
                "document_count": len(corpus.documents),
                "maximum_length": arguments.maximum_length,
                "state_dimension": state_dimension,
                "dataset_name": arguments.dataset_name,
                "dataset_shard_count": arguments.dataset_shard_count,
                "source_shard": source_shard,
                "seed": arguments.seed,
                "dtype": "float16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return np.load(state_path, mmap_mode="r")


def select_cached_states(state_cache: np.memmap, rows: Tensor) -> Tensor:
    """Copy an indexed float16 cache slice into a writable CPU tensor."""
    return torch.from_numpy(np.array(state_cache[rows.tolist()], copy=True))


def model_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


def decoder_loss_from_states(
    model: T5ForConditionalGeneration,
    states: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    pad_token_id: int,
    device: torch.device,
) -> Tensor:
    labels = input_ids.to(device).clone()
    labels[labels == pad_token_id] = -100
    output = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=states.to(device, dtype=model_dtype(model))),
        attention_mask=attention_mask.to(device),
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return output.loss


def decoder_trainable_parameters(model: T5ForConditionalGeneration) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("decoder."))
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("T5 decoder calibration found no trainable parameters")
    return trainable


def train_decoder_interface(
    model: T5ForConditionalGeneration,
    corpus: TokenizedCorpus,
    state_cache: np.memmap,
    train_rows: Tensor,
    *,
    arguments: argparse.Namespace,
    pad_token_id: int,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.AdamW(
        decoder_trainable_parameters(model),
        lr=arguments.decoder_learning_rate,
        weight_decay=0.01,
    )
    generator = torch.Generator().manual_seed(arguments.seed + 1)
    losses: list[float] = []
    model.train()
    model.encoder.eval()
    progress = tqdm(range(1, arguments.decoder_steps + 1), desc="T5 decoder calibration", unit="step", dynamic_ncols=True)
    with arguments.progress_path.open("a", encoding="utf-8") as output:
        for step in progress:
            rows = train_rows[torch.randint(len(train_rows), (arguments.batch_size,), generator=generator)]
            loss = decoder_loss_from_states(
                model,
                select_cached_states(state_cache, rows),
                corpus.input_ids[rows],
                corpus.attention_mask[rows],
                pad_token_id=pad_token_id,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_trainable_parameters(model), 1.0)
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            if step == 1 or step % arguments.log_interval == 0 or step == arguments.decoder_steps:
                progress.set_postfix(loss=f"{value:.4f}")
                output.write(json.dumps({"stage": "decoder", "step": step, "loss": value}) + "\n")
                output.flush()
    model.eval().requires_grad_(False)
    return losses


@torch.inference_mode()
def evaluate_decoder_interface(
    model: T5ForConditionalGeneration,
    corpus: TokenizedCorpus,
    state_cache: np.memmap,
    rows: Tensor,
    *,
    arguments: argparse.Namespace,
    pad_token_id: int,
    device: torch.device,
) -> float:
    losses: list[float] = []
    for batch_rows in rows.split(arguments.batch_size):
        losses.append(
            float(
                decoder_loss_from_states(
                    model,
                    select_cached_states(state_cache, batch_rows),
                    corpus.input_ids[batch_rows],
                    corpus.attention_mask[batch_rows],
                    pad_token_id=pad_token_id,
                    device=device,
                )
            )
        )
    return float(np.mean(losses))


class ContinuousSlotCodec(nn.Module):
    """An attention-resampler with a continuous T5-state decoder interface."""

    def __init__(
        self,
        *,
        state_dimension: int,
        slot_count: int,
        slot_dimension: int,
        maximum_length: int,
    ) -> None:
        super().__init__()
        self.state_dimension = state_dimension
        self.slot_count = slot_count
        self.slot_dimension = slot_dimension
        self.maximum_length = maximum_length
        self.state_to_slot = nn.Linear(state_dimension, slot_dimension)
        self.encoder_position = nn.Parameter(torch.empty(maximum_length, slot_dimension))
        self.slot_queries = nn.Parameter(torch.empty(slot_count, slot_dimension))
        self.slot_norm = nn.LayerNorm(slot_dimension)
        self.decoder_position = nn.Parameter(torch.empty(maximum_length, slot_dimension))
        self.query_projection = nn.Linear(slot_dimension, slot_dimension, bias=False)
        self.key_projection = nn.Linear(slot_dimension, slot_dimension, bias=False)
        self.value_projection = nn.Linear(slot_dimension, slot_dimension, bias=False)
        self.slot_to_state = nn.Sequential(
            nn.LayerNorm(slot_dimension),
            nn.Linear(slot_dimension, 2 * state_dimension),
            nn.GELU(),
            nn.Linear(2 * state_dimension, state_dimension),
        )
        nn.init.normal_(self.encoder_position, std=0.02)
        nn.init.normal_(self.slot_queries, std=0.02)
        nn.init.normal_(self.decoder_position, std=0.02)

    @property
    def latent_dimension(self) -> int:
        return self.slot_count * self.slot_dimension

    def encode(self, states: Tensor, attention_mask: Tensor) -> Tensor:
        token_features = self.state_to_slot(states) + self.encoder_position
        scores = torch.einsum("bld,kd->blk", token_features, self.slot_queries)
        scores = scores / self.slot_dimension**0.5
        scores = scores.masked_fill(~attention_mask.bool().unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1)
        slots = torch.einsum("blk,bld->bkd", weights, token_features)
        return rearrange(self.slot_norm(slots), "batch slots dimension -> batch (slots dimension)")

    def decode(self, latents: Tensor) -> Tensor:
        slots = rearrange(
            latents,
            "batch (slots dimension) -> batch slots dimension",
            slots=self.slot_count,
            dimension=self.slot_dimension,
        )
        queries = self.query_projection(self.decoder_position).unsqueeze(0)
        keys = self.key_projection(slots)
        values = self.value_projection(slots)
        weights = (queries @ keys.transpose(1, 2) / self.slot_dimension**0.5).softmax(dim=-1)
        return self.slot_to_state(weights @ values)

    def forward(self, states: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(states, attention_mask)
        return latents, self.decode(latents)


class MaskedSlotPredictor(nn.Module):
    """Predict full-sentence slot coordinates from a masked context encoding."""

    def __init__(self, latent_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dimension + 1),
            nn.Linear(latent_dimension + 1, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, latent_dimension),
        )

    def forward(self, context_latents: Tensor, visible_mask: Tensor, full_mask: Tensor) -> Tensor:
        visible_fraction = visible_mask.sum(1, keepdim=True).to(context_latents.dtype) / full_mask.sum(
            1, keepdim=True
        ).clamp_min(1).to(context_latents.dtype)
        return context_latents + self.network(torch.cat((context_latents, visible_fraction), dim=1))


def masked_context_mask(attention_mask: Tensor, *, probability: float, generator: torch.Generator) -> Tensor:
    sampled = torch.rand(attention_mask.shape, generator=generator)
    visible = attention_mask.bool() & (sampled >= probability)
    visible[:, 0] = attention_mask[:, 0].bool()
    return visible


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, momentum: float) -> None:
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.lerp_(source_parameter, 1.0 - momentum)


def codec_losses(
    codec: ContinuousSlotCodec,
    target_codec: ContinuousSlotCodec,
    predictor: MaskedSlotPredictor,
    model: T5ForConditionalGeneration,
    states: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    pad_token_id: int,
    masked_probability: float,
    generator: torch.Generator,
    state_weight: float,
    text_weight: float,
    cycle_weight: float,
    jepa_weight: float,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    states = states.to(device, dtype=model_dtype(model))
    masks = attention_mask.to(device)
    latents, reconstructed = codec(states, masks)
    valid = masks.unsqueeze(-1).to(reconstructed.dtype)
    state_loss = ((reconstructed - states).square() * valid).sum() / (
        valid.sum().clamp_min(1.0) * reconstructed.shape[-1]
    )
    text_loss = decoder_loss_from_states(
        model,
        reconstructed,
        input_ids,
        attention_mask,
        pad_token_id=pad_token_id,
        device=device,
    )
    latent_cycle_loss = F.mse_loss(codec.encode(reconstructed, masks), latents)
    visible = masked_context_mask(attention_mask, probability=masked_probability, generator=generator).to(device)
    context_states = states * visible.unsqueeze(-1).to(states.dtype)
    context_latents = codec.encode(context_states, visible)
    with torch.no_grad():
        target_latents = target_codec.encode(states, masks)
    jepa_loss = F.mse_loss(predictor(context_latents, visible, masks), target_latents)
    loss = (
        state_weight * state_loss
        + text_weight * text_loss
        + cycle_weight * latent_cycle_loss
        + jepa_weight * jepa_loss
    )
    return loss, {
        "state_loss": state_loss,
        "text_loss": text_loss,
        "latent_cycle_loss": latent_cycle_loss,
        "jepa_loss": jepa_loss,
    }


def train_codec(
    codec: ContinuousSlotCodec,
    predictor: MaskedSlotPredictor,
    model: T5ForConditionalGeneration,
    corpus: TokenizedCorpus,
    state_cache: np.memmap,
    train_rows: Tensor,
    *,
    arguments: argparse.Namespace,
    pad_token_id: int,
    device: torch.device,
) -> tuple[ContinuousSlotCodec, list[float]]:
    target_codec = copy.deepcopy(codec).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(codec.parameters()) + list(predictor.parameters()),
        lr=arguments.codec_learning_rate,
        weight_decay=1e-5,
    )
    generator = torch.Generator().manual_seed(arguments.seed + 2)
    losses: list[float] = []
    codec.train()
    predictor.train()
    progress = tqdm(range(1, arguments.codec_steps + 1), desc="OWT continuous slot codec", unit="step", dynamic_ncols=True)
    with arguments.progress_path.open("a", encoding="utf-8") as output:
        for step in progress:
            rows = train_rows[torch.randint(len(train_rows), (arguments.batch_size,), generator=generator)]
            loss, components = codec_losses(
                codec,
                target_codec,
                predictor,
                model,
                select_cached_states(state_cache, rows),
                corpus.input_ids[rows],
                corpus.attention_mask[rows],
                pad_token_id=pad_token_id,
                masked_probability=arguments.masked_context_probability,
                generator=generator,
                state_weight=arguments.state_loss_weight,
                text_weight=arguments.text_loss_weight,
                cycle_weight=arguments.latent_cycle_weight,
                jepa_weight=arguments.jepa_weight,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(codec.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            update_ema(target_codec, codec, arguments.ema_momentum)
            value = float(loss.detach())
            losses.append(value)
            if step == 1 or step % arguments.log_interval == 0 or step == arguments.codec_steps:
                logged = {name: float(component.detach()) for name, component in components.items()}
                progress.set_postfix(loss=f"{value:.4f}", text=f"{logged['text_loss']:.3f}", jepa=f"{logged['jepa_loss']:.3f}")
                output.write(json.dumps({"stage": "codec", "step": step, "loss": value, **logged}) + "\n")
                output.flush()
    return target_codec, losses


@torch.inference_mode()
def evaluate_codec(
    codec: ContinuousSlotCodec,
    predictor: MaskedSlotPredictor,
    target_codec: ContinuousSlotCodec,
    model: T5ForConditionalGeneration,
    corpus: TokenizedCorpus,
    state_cache: np.memmap,
    rows: Tensor,
    *,
    arguments: argparse.Namespace,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(arguments.seed + 3)
    measurements: dict[str, list[float]] = {"state_loss": [], "text_loss": [], "latent_cycle_loss": [], "jepa_loss": []}
    codec.eval()
    predictor.eval()
    for batch_rows in rows.split(arguments.batch_size):
        _, components = codec_losses(
            codec,
            target_codec,
            predictor,
            model,
            select_cached_states(state_cache, batch_rows),
            corpus.input_ids[batch_rows],
            corpus.attention_mask[batch_rows],
            pad_token_id=pad_token_id,
            masked_probability=arguments.masked_context_probability,
            generator=generator,
            state_weight=arguments.state_loss_weight,
            text_weight=arguments.text_loss_weight,
            cycle_weight=arguments.latent_cycle_weight,
            jepa_weight=arguments.jepa_weight,
            device=device,
        )
        for name, component in components.items():
            measurements[name].append(float(component))
    return {name: float(np.mean(values)) for name, values in measurements.items()}


@torch.inference_mode()
def encode_slots(
    codec: ContinuousSlotCodec,
    state_cache: np.memmap,
    attention_mask: Tensor,
    rows: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    portions: list[Tensor] = []
    codec.eval()
    for batch_rows in rows.split(batch_size):
        portions.append(
            codec.encode(
                select_cached_states(state_cache, batch_rows).to(device, dtype=next(codec.parameters()).dtype),
                attention_mask[batch_rows].to(device),
            ).cpu()
        )
    return torch.cat(portions)


@torch.inference_mode()
def decode_slots(
    codec: ContinuousSlotCodec,
    model: T5ForConditionalGeneration,
    tokenizer: object,
    latents: Tensor,
    attention_mask: Tensor,
    *,
    maximum_length: int,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    documents: list[str] = []
    for latent_batch, mask_batch in zip(latents.split(batch_size), attention_mask.split(batch_size)):
        states = codec.decode(latent_batch.to(device, dtype=next(codec.parameters()).dtype))
        generated = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=states.to(dtype=model_dtype(model))),
            attention_mask=mask_batch.to(device),
            max_new_tokens=maximum_length,
            do_sample=False,
        )
        documents.extend(tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True))
    return documents


def token_sequence(input_ids: Tensor, attention_mask: Tensor, *, pad_token_id: int) -> list[int]:
    return [
        int(token)
        for token, is_valid in zip(input_ids.tolist(), attention_mask.tolist())
        if is_valid and token != pad_token_id
    ]


def levenshtein_distance(first: list[int], second: list[int]) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_token in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_token in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_token != second_token),
                )
            )
        previous = current
    return previous[-1]


@torch.inference_mode()
def mean_pool(states: Tensor, attention_mask: Tensor) -> Tensor:
    weights = attention_mask.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(1) / weights.sum(1).clamp_min(1.0)


@torch.inference_mode()
def encode_text_features(
    model: T5ForConditionalGeneration,
    tokenizer: object,
    documents: list[str],
    *,
    maximum_length: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    tokenized = tokenizer(
        documents,
        padding="max_length",
        truncation=True,
        max_length=maximum_length,
        return_tensors="pt",
    )
    features: list[Tensor] = []
    model.encoder.eval()
    for input_batch, mask_batch in zip(tokenized["input_ids"].split(batch_size), tokenized["attention_mask"].split(batch_size)):
        states = model.encoder(
            input_ids=input_batch.to(device),
            attention_mask=mask_batch.to(device),
            return_dict=True,
        ).last_hidden_state
        features.append(mean_pool(states, mask_batch.to(device)).cpu())
    return torch.cat(features)


@torch.inference_mode()
def evaluate_decoded_reconstructions(
    codec: ContinuousSlotCodec,
    model: T5ForConditionalGeneration,
    tokenizer: object,
    corpus: TokenizedCorpus,
    state_cache: np.memmap,
    validation_rows: Tensor,
    *,
    arguments: argparse.Namespace,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, float]:
    rows = validation_rows[: arguments.decode_evaluation_count]
    latents = encode_slots(
        codec,
        state_cache,
        corpus.attention_mask,
        rows,
        batch_size=arguments.batch_size,
        device=device,
    )
    decoded = decode_slots(
        codec,
        model,
        tokenizer,
        latents,
        corpus.attention_mask[rows],
        maximum_length=arguments.maximum_length,
        batch_size=arguments.batch_size,
        device=device,
    )
    decoded_tokens = tokenizer(
        decoded,
        padding="max_length",
        truncation=True,
        max_length=arguments.maximum_length,
        return_tensors="pt",
    )
    edit_distances = []
    for row, generated_ids, generated_mask in zip(rows, decoded_tokens["input_ids"], decoded_tokens["attention_mask"]):
        source = token_sequence(corpus.input_ids[row], corpus.attention_mask[row], pad_token_id=pad_token_id)
        generated = token_sequence(generated_ids, generated_mask, pad_token_id=pad_token_id)
        edit_distances.append(levenshtein_distance(source, generated) / max(len(source), len(generated), 1))
    source_states = select_cached_states(state_cache, rows).float()
    source_features = mean_pool(source_states, corpus.attention_mask[rows])
    decoded_features = encode_text_features(
        model,
        tokenizer,
        decoded,
        maximum_length=arguments.maximum_length,
        batch_size=arguments.batch_size,
        device=device,
    )
    similarities = F.normalize(decoded_features, dim=1) @ F.normalize(source_features, dim=1).T
    retrieval = float((similarities.argmax(1) == torch.arange(len(rows))).float().mean())
    examples_path = arguments.output_dir / "heldout_decoded_examples.jsonl"
    examples_path.write_text(
        "".join(
            json.dumps({"source": corpus.documents[int(row)], "decoded_reconstruction": reconstructed}) + "\n"
            for row, reconstructed in zip(rows, decoded)
        ),
        encoding="utf-8",
    )
    return {
        "decoded_example_count": len(rows),
        "mean_normalized_token_edit_distance": float(np.mean(edit_distances)),
        "semantic_retrieval_top1": retrieval,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="Skylion007/openwebtext")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--data-cache-root", type=Path, default=Path("data/huggingface"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text_owt_t5_slot_codec_50k"))
    parser.add_argument("--t5-model", default="t5-small")
    parser.add_argument("--maximum-length", type=int, default=32)
    parser.add_argument("--train-document-count", type=int, default=50_000)
    parser.add_argument("--validation-document-count", type=int, default=1_000)
    parser.add_argument("--minimum-document-characters", type=int, default=40)
    parser.add_argument("--dataset-shard-count", type=int, default=80)
    parser.add_argument("--source-shard", type=int, default=-1)
    parser.add_argument("--shuffle-buffer", type=int, default=20_000)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decoder-steps", type=int, default=2_500)
    parser.add_argument("--decoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--slot-count", type=int, default=32)
    parser.add_argument("--slot-dimension", type=int, default=128)
    parser.add_argument("--codec-steps", type=int, default=20_000)
    parser.add_argument("--codec-learning-rate", type=float, default=3e-4)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=0.25)
    parser.add_argument("--latent-cycle-weight", type=float, default=0.10)
    parser.add_argument("--jepa-weight", type=float, default=0.10)
    parser.add_argument("--masked-context-probability", type=float, default=0.35)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--predictor-hidden-dimension", type=int, default=2_048)
    parser.add_argument("--decode-evaluation-count", type=int, default=128)
    parser.add_argument("--codec-ce-gap-gate", type=float, default=0.50)
    parser.add_argument("--edit-distance-gate", type=float, default=0.20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if min(arguments.train_document_count, arguments.validation_document_count, arguments.batch_size) < 1:
        raise ValueError("document counts and batch size must be positive")
    if arguments.train_document_count < arguments.batch_size:
        raise ValueError("train_document_count must be at least batch_size")
    if min(arguments.decoder_steps, arguments.codec_steps, arguments.slot_count, arguments.slot_dimension) < 1:
        raise ValueError("training steps and slot dimensions must be positive")
    if not 0.0 < arguments.masked_context_probability < 1.0:
        raise ValueError("masked_context_probability must lie in (0, 1)")
    if not 0.0 < arguments.ema_momentum < 1.0:
        raise ValueError("ema_momentum must lie in (0, 1)")
    if arguments.dataset_shard_count < 1:
        raise ValueError("dataset_shard_count must be positive")
    if arguments.source_shard >= arguments.dataset_shard_count:
        raise ValueError("source_shard must be smaller than dataset_shard_count")
    if arguments.decode_evaluation_count > arguments.validation_document_count:
        raise ValueError("decode_evaluation_count must not exceed validation_document_count")


def run(arguments: argparse.Namespace) -> dict[str, object]:
    validate_arguments(arguments)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    arguments.data_cache_root.mkdir(parents=True, exist_ok=True)
    arguments.progress_path = arguments.output_dir / "training_progress.jsonl"
    arguments.progress_path.write_text("", encoding="utf-8")
    torch.manual_seed(arguments.seed)
    device = device_from_argument(arguments.device)
    documents = stream_openwebtext_documents(arguments)
    tokenizer = AutoTokenizer.from_pretrained(arguments.t5_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("T5 tokenizer must provide a pad token")
    corpus = tokenize_documents(tokenizer, documents, maximum_length=arguments.maximum_length)
    model = T5ForConditionalGeneration.from_pretrained(arguments.t5_model, local_files_only=True).to(device)
    state_cache = cache_frozen_encoder_states(model, corpus, arguments=arguments, device=device)
    train_rows = torch.arange(arguments.train_document_count)
    validation_rows = torch.arange(arguments.train_document_count, len(documents))
    started_at = time.perf_counter()
    decoder_losses = train_decoder_interface(
        model,
        corpus,
        state_cache,
        train_rows,
        arguments=arguments,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    native_validation_ce = evaluate_decoder_interface(
        model,
        corpus,
        state_cache,
        validation_rows,
        arguments=arguments,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    codec = ContinuousSlotCodec(
        state_dimension=model.config.d_model,
        slot_count=arguments.slot_count,
        slot_dimension=arguments.slot_dimension,
        maximum_length=arguments.maximum_length,
    ).to(device)
    predictor = MaskedSlotPredictor(codec.latent_dimension, arguments.predictor_hidden_dimension).to(device)
    target_codec, codec_losses_history = train_codec(
        codec,
        predictor,
        model,
        corpus,
        state_cache,
        train_rows,
        arguments=arguments,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    codec_metrics = evaluate_codec(
        codec,
        predictor,
        target_codec,
        model,
        corpus,
        state_cache,
        validation_rows,
        arguments=arguments,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    decoded_metrics = evaluate_decoded_reconstructions(
        codec,
        model,
        tokenizer,
        corpus,
        state_cache,
        validation_rows,
        arguments=arguments,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    state_path = arguments.output_dir / "owt_t5_slot_codec.pt"
    torch.save(
        {
            "codec": codec.state_dict(),
            "predictor": predictor.state_dict(),
            "target_codec": target_codec.state_dict(),
            "configuration": vars(arguments),
        },
        state_path,
    )
    codec_ce_gap = codec_metrics["text_loss"] - native_validation_ce
    gates = {
        "codec_ce_gap_at_most_0_5": codec_ce_gap <= arguments.codec_ce_gap_gate,
        "normalized_token_edit_distance_at_most_0_2": decoded_metrics["mean_normalized_token_edit_distance"] <= arguments.edit_distance_gate,
    }
    gates["all_pass"] = all(gates.values())
    summary = {
        "stage": "Stage 1: continuous OWT codec pretraining",
        "method": "frozen T5 boundary + continuous slots + masked-context joint embedding prediction",
        "dataset": arguments.dataset_name,
        "split": arguments.dataset_split,
        "dataset_shard_count": arguments.dataset_shard_count,
        "source_shard": arguments.source_shard if arguments.source_shard >= 0 else arguments.seed % arguments.dataset_shard_count,
        "train_document_count": arguments.train_document_count,
        "validation_document_count": arguments.validation_document_count,
        "representation": {
            "native_state_shape": [arguments.maximum_length, model.config.d_model],
            "slot_shape": [arguments.slot_count, arguments.slot_dimension],
            "compression_ratio": (arguments.maximum_length * model.config.d_model) / codec.latent_dimension,
        },
        "uses_flow_matching": False,
        "uses_rectified_flow": False,
        "uses_diffusion_sampling": False,
        "uses_ode_solver": False,
        "decoder_final_loss": decoder_losses[-1],
        "native_validation_cross_entropy": native_validation_ce,
        "codec_final_loss": codec_losses_history[-1],
        "codec_validation": codec_metrics,
        "decoded_validation": decoded_metrics,
        "codec_ce_gap": codec_ce_gap,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    (arguments.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), indent=2))
