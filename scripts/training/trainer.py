"""PIQA training loop.

The caller is responsible for the optimizer and checkpointing.
The loss function is fixed to CrossEntropyLoss.
train() updates only the supplied model, keeps the parameters from the epoch
with the best validation accuracy in memory, and restores them before returning.
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.evaluation.evaluate import evaluate

SEED = 4012


def set_seed(seed: int = SEED) -> None:
    """Set the random seeds for random, NumPy, PyTorch, and CUDA.

    Call this function before creating the model.

    To make the batch order reproducible across training epochs, the caller
    must separately initialize a torch.Generator for the DataLoader or
    BatchSampler, seed it with manual_seed(), and pass it in.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _logits_of(output):
    """Extract logits from model output.

    RNNWithAttention returns (logits, weights), whereas
    TransformerClassifier returns only logits.
    """
    return output[0] if isinstance(output, (tuple, list)) else output


def _to(batch, device):
    """Move a batch of inputs, masks, and labels to the target device."""
    input_1, input_2, mask_1, mask_2, labels = batch
    return (
        input_1.to(device, non_blocking=True),
        input_2.to(device, non_blocking=True),
        mask_1.to(device, non_blocking=True),
        mask_2.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    max_epochs,
    patience=None,
    grad_clip=None,
    train_eval_loader=None,
    verbose=True,
):
    """
    Train for up to max_epochs epochs and evaluate on the validation set after each epoch.

    The loss function is fixed to nn.CrossEntropyLoss() with mean reduction.
    The batch loss is multiplied by the batch size and then divided by the
    total number of examples at the end of the epoch. This uses the same
    averaging convention as evaluate(), which computes total loss divided
    by the number of examples.

    If patience is an integer, training stops when the validation accuracy
    has failed to improve for that many consecutive epochs.
    If patience is None, training continues for all max_epochs.

    Whether or not early stopping occurs, the returned model is restored to
    the parameters from the epoch with the highest validation accuracy.
    This function does not write any files.
    """
    device = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    max_norm = float("inf") if grad_clip is None else float(grad_clip)

    rows = []
    best_acc = -float("inf")
    best_state = None
    stall = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()

        total_loss = torch.zeros((), device=device)
        total_correct = torch.zeros((), device=device)
        grad_sum = torch.zeros((), device=device)
        grad_max = torch.zeros((), device=device)
        n_seen = 0
        n_batches = 0

        for batch in train_loader:
            input_1, input_2, mask_1, mask_2, labels = _to(batch, device)
            logits = _logits_of(model(input_1, input_2, mask_1, mask_2))
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.detach() * batch_size
            total_correct += (logits.detach().argmax(dim=1) == labels).sum()
            grad_sum += grad_norm.detach()
            grad_max = torch.maximum(grad_max, grad_norm.detach())
            n_seen += batch_size
            n_batches += 1

        # Synchronize only once per epoch to avoid blocking the GPU on every batch.
        stats = torch.stack([
            total_loss / n_seen,
            total_correct / n_seen,
            grad_sum / max(n_batches, 1),
            grad_max,
        ]).tolist()
        train_loss, train_accuracy, grad_norm_mean, grad_norm_max = stats

        valid_metrics, _ = evaluate(model, val_loader, device)
        if train_eval_loader is None:
            train_eval_accuracy = None
        else:
            train_eval_metrics, _ = evaluate(model, train_eval_loader, device)
            train_eval_accuracy = train_eval_metrics["accuracy"]

        valid_accuracy = valid_metrics["accuracy"]
        is_best = valid_accuracy > best_acc
        if is_best:
            best_acc = valid_accuracy
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
        else:
            stall += 1

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_eval_accuracy": train_eval_accuracy,
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_accuracy,
            "valid_precision": valid_metrics["precision"],
            "valid_recall": valid_metrics["recall"],
            "valid_f1": valid_metrics["f1"],
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_max": grad_norm_max,
            "is_best": bool(is_best),
            "epoch_seconds": round(time.perf_counter() - epoch_start, 2),
        })

        if verbose:
            mark = "  * best" if is_best else ""
            eval_text = (
                "" if train_eval_accuracy is None
                else f" | Train-eval Acc: {train_eval_accuracy: .3f}"
            )
            print(
                f"Epochs: {epoch} | Train Loss: {train_loss: .3f} "
                f"| Train Accuracy: {train_accuracy: .3f}{eval_text} "
                f"| Val Loss: {valid_metrics['loss']: .3f} "
                f"| Val Accuracy: {valid_accuracy: .3f}{mark}"
            )

        if patience is not None and stall >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(rows, columns=[
        "epoch", "train_loss", "train_accuracy", "train_eval_accuracy",
        "valid_loss", "valid_accuracy", "valid_precision", "valid_recall",
        "valid_f1", "grad_norm_mean", "grad_norm_max", "is_best", "epoch_seconds",
    ])
    return model, history
