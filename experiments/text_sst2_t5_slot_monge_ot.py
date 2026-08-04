"""Static K=1 Monge transport in compact continuous T5 sentence slots.

Sentences are represented by frozen T5 encoder states, then compressed from a
``maximum_length x 512`` sequence into ``slot_count x slot_dimension``
continuous coordinates.  A calibrated T5 decoder reads the expanded state.
The transport itself is the exact-inverse atlas

    T[a,b](z) = H^-1(P_b^-1(P_a(s)) || r),

where ``H`` is an orthogonal chart, ``s`` a small style coordinate block,
``r`` retained coordinates, and ``P`` label-conditioned invertible affine
couplings.  Finite equal-mass Hungarian assignments to a common barycenter
supervise the canonicalizers.  Held-out endpoints use only a held-out source
slot state and a requested label.

This is static optimal transport: no velocity field, flow-matching loss,
rectified flow, diffusion sampler, or ODE solver is used.  T5 is only a frozen
continuous encoder/decoder interface; its token simplex is never transported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from text_sst2_slot_monge_ot import (
    LABEL_NAMES,
    LabelProbe,
    LatentCoordinates,
    SlotMongeAtlas,
    device_from_argument,
    discrete_barycenter,
    exact_squared_cost,
    render_transport_pca,
    select_balanced_examples,
    style_supports,
    train_canonicalizers,
    train_chart,
    train_semantic_probe,
    transport_batches,
)
from text_sst2_t5_monge_ot import (
    SST2Example,
    calibrate_decoder,
    decoder_loss,
    encode_examples,
    ensure_sst2,
    evaluate_decoder_loss,
    load_sst2_split,
)


class SlotStateAutoencoder(nn.Module):
    """Attention-resampler compressor and cross-attention state expander."""

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
        self.state_to_slot = nn.Linear(self.state_dimension, self.slot_dimension)
        self.slot_queries = nn.Parameter(torch.empty(self.slot_count, self.slot_dimension))
        self.slot_norm = nn.LayerNorm(self.slot_dimension)
        self.position_queries = nn.Parameter(torch.empty(self.maximum_length, self.slot_dimension))
        self.query_projection = nn.Linear(self.slot_dimension, self.slot_dimension, bias=False)
        self.key_projection = nn.Linear(self.slot_dimension, self.slot_dimension, bias=False)
        self.value_projection = nn.Linear(self.slot_dimension, self.slot_dimension, bias=False)
        self.slot_to_state = nn.Sequential(
            nn.LayerNorm(self.slot_dimension),
            nn.Linear(self.slot_dimension, 2 * self.state_dimension),
            nn.GELU(),
            nn.Linear(2 * self.state_dimension, self.state_dimension),
        )
        nn.init.normal_(self.slot_queries, std=0.02)
        nn.init.normal_(self.position_queries, std=0.02)

    @property
    def latent_dimension(self) -> int:
        return self.slot_count * self.slot_dimension

    def encode(self, states: Tensor, attention_mask: Tensor) -> Tensor:
        token_features = self.state_to_slot(states)
        scores = torch.einsum("bld,kd->blk", token_features, self.slot_queries)
        scores = scores.masked_fill(~attention_mask.bool().unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1)
        slots = torch.einsum("blk,bld->bkd", weights, token_features)
        return self.slot_norm(slots).flatten(1)

    def decode(self, latents: Tensor) -> Tensor:
        slots = latents.unflatten(1, (self.slot_count, self.slot_dimension))
        queries = self.query_projection(self.position_queries).unsqueeze(0)
        keys = self.key_projection(slots)
        values = self.value_projection(slots)
        weights = (queries @ keys.transpose(1, 2) / self.slot_dimension**0.5).softmax(dim=-1)
        expanded = weights @ values
        return self.slot_to_state(expanded)

    def forward(self, states: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(states, attention_mask)
        return latents, self.decode(latents)


@torch.inference_mode()
def encode_slot_batches(
    model: SlotStateAutoencoder,
    states: Tensor,
    attention_mask: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    portions: list[Tensor] = []
    for state_batch, mask_batch in zip(states.split(batch_size), attention_mask.split(batch_size)):
        portions.append(model.encode(state_batch.to(device), mask_batch.to(device)).cpu())
    return torch.cat(portions)


def slot_autoencoder_loss(
    slots: SlotStateAutoencoder,
    model: T5ForConditionalGeneration,
    states: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    pad_token_id: int,
    state_weight: float,
    text_weight: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    latents, reconstructed = slots(states.to(device), attention_mask.to(device))
    valid = attention_mask.to(device).unsqueeze(-1).to(reconstructed.dtype)
    state_loss = ((reconstructed - states.to(device)).square() * valid).sum() / (
        valid.sum().clamp_min(1.0) * reconstructed.shape[-1]
    )
    text_loss = decoder_loss(
        model,
        reconstructed,
        input_ids,
        attention_mask,
        pad_token_id=pad_token_id,
        device=device,
    )
    return state_weight * state_loss + text_weight * text_loss, state_loss, text_loss


def train_slot_autoencoder(
    slots: SlotStateAutoencoder,
    model: T5ForConditionalGeneration,
    states: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    pad_token_id: int,
    configuration: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.AdamW(slots.parameters(), lr=configuration.slot_learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(configuration.seed + 101)
    losses: list[float] = []
    progress = tqdm(range(1, configuration.slot_steps + 1), desc="Continuous T5 slot autoencoder", unit="step", dynamic_ncols=True)
    with configuration.progress_path.open("a", encoding="utf-8") as output:
        for step in progress:
            rows = torch.randint(len(states), (configuration.batch_size,), generator=generator)
            loss, state_loss, text_loss = slot_autoencoder_loss(
                slots,
                model,
                states[rows],
                input_ids[rows],
                attention_mask[rows],
                pad_token_id=pad_token_id,
                state_weight=configuration.state_loss_weight,
                text_weight=configuration.text_loss_weight,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(slots.parameters(), 1.0)
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            if step == 1 or step % configuration.log_interval == 0 or step == configuration.slot_steps:
                state_value = float(state_loss.detach())
                text_value = float(text_loss.detach())
                progress.set_postfix(loss=f"{value:.4f}", state=f"{state_value:.3f}", text=f"{text_value:.3f}")
                output.write(json.dumps({"stage": "slot_autoencoder", "step": step, "loss": value, "state_loss": state_value, "text_loss": text_value}) + "\n")
                output.flush()
    return losses


@torch.inference_mode()
def evaluate_slot_autoencoder(
    slots: SlotStateAutoencoder,
    model: T5ForConditionalGeneration,
    states: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    pad_token_id: int,
    configuration: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    state_losses: list[float] = []
    text_losses: list[float] = []
    for state_batch, input_batch, mask_batch in zip(
        states.split(configuration.batch_size),
        input_ids.split(configuration.batch_size),
        attention_mask.split(configuration.batch_size),
    ):
        _, reconstructed = slots(state_batch.to(device), mask_batch.to(device))
        valid = mask_batch.to(device).unsqueeze(-1).to(reconstructed.dtype)
        state_losses.append(float(((reconstructed - state_batch.to(device)).square() * valid).sum() / (valid.sum().clamp_min(1.0) * reconstructed.shape[-1])))
        text_losses.append(float(decoder_loss(model, reconstructed, input_batch, mask_batch, pad_token_id=pad_token_id, device=device)))
    return {"state_mse": float(sum(state_losses) / len(state_losses)), "text_cross_entropy": float(sum(text_losses) / len(text_losses))}


@torch.inference_mode()
def decode_slot_latents(
    slots: SlotStateAutoencoder,
    model: T5ForConditionalGeneration,
    tokenizer: object,
    latents: Tensor,
    source_mask: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    decoded: list[str] = []
    for latent_batch, mask_batch in zip(latents.split(batch_size), source_mask.split(batch_size)):
        states = slots.decode(latent_batch.to(device))
        generated = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=states),
            attention_mask=mask_batch.to(device),
            max_new_tokens=slots.maximum_length,
            do_sample=False,
        )
        decoded.extend(tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True))
    return decoded


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/sst2"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text_sst2_t5_slot_monge_ot"))
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--t5-model", default="t5-small")
    parser.add_argument("--maximum-length", type=int, default=32)
    parser.add_argument("--autoencoder-count-per-label", type=int, default=1_024)
    parser.add_argument("--training-count-per-label", type=int, default=512)
    parser.add_argument("--validation-count-per-label", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decoder-steps", type=int, default=750)
    parser.add_argument("--decoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--slot-count", type=int, default=8)
    parser.add_argument("--slot-dimension", type=int, default=128)
    parser.add_argument("--style-slot-count", type=int, default=2)
    parser.add_argument("--slot-steps", type=int, default=2_000)
    parser.add_argument("--slot-learning-rate", type=float, default=3e-4)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=0.25)
    parser.add_argument("--chart-steps", type=int, default=1_000)
    parser.add_argument("--transport-steps", type=int, default=2_000)
    parser.add_argument("--chart-learning-rate", type=float, default=2e-4)
    parser.add_argument("--transport-learning-rate", type=float, default=2e-4)
    parser.add_argument("--chart-reflections", type=int, default=64)
    parser.add_argument("--atlas-hidden-dimension", type=int, default=256)
    parser.add_argument("--canonicalizer-layers", type=int, default=4)
    parser.add_argument("--log-scale-limit", type=float, default=0.25)
    parser.add_argument("--translation-limit", type=float, default=0.50)
    parser.add_argument("--style-weight", type=float, default=2.0)
    parser.add_argument("--adversarial-weight", type=float, default=0.10)
    parser.add_argument("--adversarial-scale", type=float, default=0.5)
    parser.add_argument("--chart-regularization-weight", type=float, default=0.03)
    parser.add_argument("--barycenter-iterations", type=int, default=3)
    parser.add_argument("--decode-examples", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=53)
    return parser.parse_args()


def validate_configuration(configuration: argparse.Namespace) -> None:
    if configuration.autoencoder_count_per_label < configuration.training_count_per_label:
        raise ValueError("autoencoder support must cover the OT training support")
    if configuration.training_count_per_label < configuration.batch_size:
        raise ValueError("training_count_per_label must be at least batch_size")
    if not 0 < configuration.style_slot_count < configuration.slot_count:
        raise ValueError("style_slot_count must be a proper non-empty slot subset")
    if configuration.style_slot_count * configuration.slot_dimension % 2:
        raise ValueError("the transported style coordinate dimension must be even")
    if min(configuration.decoder_steps, configuration.slot_steps, configuration.chart_steps, configuration.transport_steps) < 1:
        raise ValueError("all training step counts must be positive")


def run(configuration: argparse.Namespace) -> dict[str, object]:
    validate_configuration(configuration)
    configuration.output_dir.mkdir(parents=True, exist_ok=True)
    configuration.progress_path = configuration.output_dir / "training_progress.jsonl"
    configuration.progress_path.write_text("", encoding="utf-8")
    torch.manual_seed(configuration.seed)
    device = device_from_argument(configuration.device)
    ensure_sst2(configuration.data_root, download=configuration.download)
    train_examples = load_sst2_split(configuration.data_root, "train")
    dev_examples = load_sst2_split(configuration.data_root, "dev")
    autoencoder_by_label = select_balanced_examples(train_examples, count_per_label=configuration.autoencoder_count_per_label, seed=configuration.seed)
    transport_by_label = {label: autoencoder_by_label[label][: configuration.training_count_per_label] for label in range(2)}
    heldout_by_label = select_balanced_examples(dev_examples, count_per_label=configuration.validation_count_per_label, seed=configuration.seed + 1)
    autoencoder_examples = autoencoder_by_label[0] + autoencoder_by_label[1]
    tokenizer = AutoTokenizer.from_pretrained(configuration.t5_model, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(configuration.t5_model, local_files_only=True).to(device)
    if tokenizer.pad_token_id is None:
        raise ValueError("T5 tokenizer must provide a pad token")
    encoded_autoencoder = encode_examples(model, tokenizer, autoencoder_examples, maximum_length=configuration.maximum_length, batch_size=configuration.batch_size, device=device)
    encoded_transport = [
        encode_examples(model, tokenizer, transport_by_label[label], maximum_length=configuration.maximum_length, batch_size=configuration.batch_size, device=device)
        for label in range(2)
    ]
    encoded_heldout = [
        encode_examples(model, tokenizer, heldout_by_label[label], maximum_length=configuration.maximum_length, batch_size=configuration.batch_size, device=device)
        for label in range(2)
    ]
    started_at = time.perf_counter()
    decoder_losses = calibrate_decoder(
        model,
        encoded_autoencoder,
        pad_token_id=tokenizer.pad_token_id,
        steps=configuration.decoder_steps,
        batch_size=configuration.batch_size,
        learning_rate=configuration.decoder_learning_rate,
        seed=configuration.seed + 10,
        device=device,
        progress_path=configuration.progress_path,
    )
    model.requires_grad_(False).eval()
    slots = SlotStateAutoencoder(
        state_dimension=encoded_autoencoder.states.shape[-1],
        slot_count=configuration.slot_count,
        slot_dimension=configuration.slot_dimension,
        maximum_length=configuration.maximum_length,
    ).to(device)
    slot_losses = train_slot_autoencoder(
        slots,
        model,
        encoded_autoencoder.states,
        encoded_autoencoder.input_ids,
        encoded_autoencoder.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        configuration=configuration,
        device=device,
    )
    slots.eval().requires_grad_(False)
    heldout_metrics = [
        evaluate_slot_autoencoder(
            slots, model, encoded.states, encoded.input_ids, encoded.attention_mask,
            pad_token_id=tokenizer.pad_token_id, configuration=configuration, device=device,
        )
        for encoded in encoded_heldout
    ]
    train_latents = [encode_slot_batches(slots, encoded.states, encoded.attention_mask, batch_size=configuration.batch_size, device=device) for encoded in encoded_transport]
    heldout_latents = [encode_slot_batches(slots, encoded.states, encoded.attention_mask, batch_size=configuration.batch_size, device=device) for encoded in encoded_heldout]
    coordinates = LatentCoordinates.from_latents(torch.cat(train_latents))
    supports = torch.stack([coordinates.normalize(latents) for latents in train_latents])
    heldout_supports = torch.stack([coordinates.normalize(latents) for latents in heldout_latents])
    atlas = SlotMongeAtlas(configuration).to(device)
    chart_losses = train_chart(atlas, supports, configuration=configuration, device=device)
    styles = style_supports(atlas, supports, batch_size=configuration.batch_size, device=device)
    barycenter, assignments, barycenter_objectives = discrete_barycenter(styles, iterations=configuration.barycenter_iterations)
    transport_losses = train_canonicalizers(atlas, styles, barycenter, assignments, configuration=configuration, device=device)
    probe_features = atlas.chart(supports.flatten(0, 1).to(device)).cpu()
    probe = train_semantic_probe(probe_features, torch.arange(2).repeat_interleave(configuration.training_count_per_label), seed=configuration.seed + 50, device=device)
    records: list[dict[str, object]] = []
    decoded_path = configuration.output_dir / "heldout_decoded_examples.jsonl"
    decoded_path.write_text("", encoding="utf-8")
    for source_label, target_label in ((0, 1), (1, 0)):
        source = heldout_supports[source_label]
        target = heldout_supports[target_label]
        endpoint = transport_batches(atlas, source, source_label=source_label, target_label=target_label, batch_size=configuration.batch_size, device=device)
        reverse = transport_batches(atlas, endpoint, source_label=target_label, target_label=source_label, batch_size=configuration.batch_size, device=device)
        predicted = probe(atlas.chart(endpoint.to(device))).argmax(1).cpu()
        direction = f"{LABEL_NAMES[source_label]}_to_{LABEL_NAMES[target_label]}"
        source_cost = exact_squared_cost(source, target)
        endpoint_cost = exact_squared_cost(endpoint, target)
        records.append({
            "source_label": LABEL_NAMES[source_label],
            "target_label": LABEL_NAMES[target_label],
            "source_to_target_ot_cost": source_cost,
            "pure_terminal_to_target_ot_cost": endpoint_cost,
            "pure_ot_improvement": source_cost - endpoint_cost,
            "pure_target_probe_rate": float((predicted == target_label).float().mean()),
            "pure_cycle_coordinate_mse": float(F.mse_loss(reverse, source)),
        })
        render_transport_pca(source, target, endpoint, source_name=LABEL_NAMES[source_label], target_name=LABEL_NAMES[target_label], output_path=configuration.output_dir / f"{direction}_pca_3d.png")
        count = min(configuration.decode_examples, len(source))
        source_text = decode_slot_latents(slots, model, tokenizer, coordinates.denormalize(source[:count]), encoded_heldout[source_label].attention_mask[:count], batch_size=configuration.batch_size, device=device)
        endpoint_text = decode_slot_latents(slots, model, tokenizer, coordinates.denormalize(endpoint[:count]), encoded_heldout[source_label].attention_mask[:count], batch_size=configuration.batch_size, device=device)
        with decoded_path.open("a", encoding="utf-8") as output:
            for example, reconstruction, generated in zip(heldout_by_label[source_label][:count], source_text, endpoint_text):
                output.write(json.dumps({"direction": direction, "source_text": example.text, "source_reconstruction": reconstruction, "pure_static_atlas": generated}) + "\n")
    torch.save({"slot_autoencoder": slots.state_dict(), "atlas": atlas.state_dict(), "coordinates": coordinates, "configuration": vars(configuration)}, configuration.output_dir / "t5_slot_monge_atlas.pt")
    summary = {
        "method": "static K=1 empirical Monge atlas in compact continuous T5 slots",
        "transport_representation": {"native_state_shape": [configuration.maximum_length, encoded_autoencoder.states.shape[-1]], "slot_shape": [configuration.slot_count, configuration.slot_dimension], "compression_ratio": (configuration.maximum_length * encoded_autoencoder.states.shape[-1]) / (configuration.slot_count * configuration.slot_dimension)},
        "uses_flow_matching": False,
        "uses_rectified_flow": False,
        "uses_diffusion_sampling": False,
        "uses_ode_solver": False,
        "heldout_target_seen_by_map": False,
        "exact_architectural_inverse": True,
        "decoder_final_loss": decoder_losses[-1],
        "decoder_heldout_cross_entropy": [evaluate_decoder_loss(model, encoded, pad_token_id=tokenizer.pad_token_id, batch_size=configuration.batch_size, device=device) for encoded in encoded_heldout],
        "slot_final_loss": slot_losses[-1],
        "slot_heldout": heldout_metrics,
        "chart_final_loss": chart_losses[-1],
        "canonicalizer_final_loss": transport_losses[-1],
        "barycenter_objectives": barycenter_objectives,
        "directions": records,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    (configuration.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), indent=2))
