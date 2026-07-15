"""Classifier training kept separate from transport estimation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .models import DinoCatDogClassifier


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    validation_accuracy: list[float]


def symmetric_prediction_kl(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    return 0.5 * (
        torch.sum(log_a.exp() * (log_a - log_b), dim=1).mean()
        + torch.sum(log_b.exp() * (log_b - log_a), dim=1).mean()
    )


@torch.inference_mode()
def evaluate_classifier(
    model: DinoCatDogClassifier,
    loader: DataLoader,
    preprocess: Callable[..., Tensor],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = total = 0
    for images, labels, _ in loader:
        images = preprocess(images.to(device, non_blocking=True), training=False)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if not isinstance(logits, Tensor):
            raise TypeError("classifier unexpectedly returned auxiliary values")
        loss_sum += float(F.cross_entropy(logits, labels)) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return loss_sum / total, correct / total


def train_classifier(
    model: DinoCatDogClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    preprocess: Callable[..., Tensor],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    consistency_weight: float = 0.25,
    gate_balance_weight: float = 0.01,
) -> TrainingHistory:
    """Fine-tune DINO with two-view consistency, retaining the best epoch."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = TrainingHistory([], [], [])
    best_accuracy = -1.0
    best_state: dict[str, Tensor] | None = None

    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for raw_images, labels, _ in train_loader:
            raw_images = raw_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            view_a = preprocess(raw_images, training=True)
            view_b = preprocess(raw_images, training=True)
            optimizer.zero_grad(set_to_none=True)
            logits_a, gate_a = model(view_a, return_gate=True)
            logits_b, gate_b = model(view_b, return_gate=True)
            classification = 0.5 * (
                F.cross_entropy(logits_a, labels, label_smoothing=0.1)
                + F.cross_entropy(logits_b, labels, label_smoothing=0.1)
            )
            loss = classification + consistency_weight * symmetric_prediction_kl(
                logits_a, logits_b
            )
            gates = [gate for gate in (gate_a, gate_b) if gate is not None]
            if gates:
                gate_values = torch.cat(gates)
                loss = loss + gate_balance_weight * (
                    gate_values.mean(0) - 0.5
                ).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
        scheduler.step()

        validation_loss, accuracy = evaluate_classifier(
            model, validation_loader, preprocess, device
        )
        history.train_loss.append(loss_sum / count)
        history.validation_loss.append(validation_loss)
        history.validation_accuracy.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return history
