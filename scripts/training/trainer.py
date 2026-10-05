"""Training and validation loop for the PIQA model comparison.

Provides the epoch loop, the optimiser step, per-epoch logging and checkpointing.
Evaluation metrics and predictions come from scripts/evaluation/evaluate.py, which is
called once per epoch on the validation loader.

Design notes
------------
    1. Fixed random seed 4012 for Python, NumPy and torch, so both models see the same
       batch order.
    2. Returns a pandas DataFrame with one row per epoch, and writes it as a CSV.
    3. Saves state_dict rather than the model object. Writing `best_model = model`
       would bind a reference, not a copy: once training continues, `best_model`
       changes with it and no snapshot is kept.
    4. Reports precision / recall / F1 alongside accuracy, from compute_metrics.
    5. Records acc_gap and the gradient norms, for the overfitting and learning-rate
       checks agreed for the comparison.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

SEED = 4012

# evaluate() returns these five; the same names are used for the training side.
_KEYS = ("loss", "accuracy", "precision", "recall", "f1")


def set_seed(seed: int = SEED) -> None:
    """Seed every random source. Call this BEFORE building the DataLoaders.

    When the train DataLoader is created with shuffle=True and no generator of its own,
    its batch order comes from the global torch RNG. Seeding after the loaders exist
    therefore has no effect on the order, and two models no longer see the same batches.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



#  Timing 
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    """Elapsed time so far, and an estimate of the time remaining."""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



#  Helpers
def _unpack(batch, device):
    """A PIQA batch is (input_1, input_2, mask_1, mask_2, labels).

    Moves all five tensors onto the training device, so the model and its inputs are
    always on the same one.
    """
    input_1, input_2, mask_1, mask_2, labels = batch
    return (input_1.to(device), input_2.to(device),
            mask_1.to(device), mask_2.to(device), labels.to(device))


def _logits_of(output):
    """RNNWithAttention returns (logits, weights); TransformerClassifier returns logits."""
    return output[0] if isinstance(output, (tuple, list)) else output


def _metrics(d) -> Dict[str, Optional[float]]:
    """Read the five metric keys out of what evaluate() or compute_metrics() returned."""
    d = d or {}
    return {k: (float(d[k]) if k in d else None) for k in _KEYS}


def _save_predictions(predictions, stem: Path, epoch: Optional[int]) -> None:
    """Write the predictions DataFrame next to the checkpoint.

    Path.with_suffix() must not be used here: in a run id such as "m3_rnn_lr0.001",
    Python reads ".001_best_predictions" as the suffix, so every learning rate would
    collapse to the same file name and the runs would overwrite each other.
    """
    if predictions is None:
        return
    df = predictions.copy()
    if epoch is not None:
        df.insert(0, "epoch", epoch)
    df.to_csv(stem.parent / f"{stem.name}.csv", index=False)



#  Training
def train(model, train_dataloader, val_dataloader, params: Dict,
          evaluate_fn: Callable,
          compute_metrics_fn: Callable,
          test_dataloader=None,
          model_name: str = "model",
          out_dir: str = "runs",
          run_id: Optional[str] = None,
          verbose: bool = True):
    """Train for params['epochs'] epochs, validating once per epoch.

    Parameters
        model               RNNWithAttention or TransformerClassifier
        train_dataloader    train_loader from get_piqa_dataloaders()
        val_dataloader      validate_loader from the same call
        params              the config dict loaded from config/*.json. Both models must
                            be given the same one. learning_rate and epochs are read
                            from it, grad_clip is optional, and the whole dict is stored
                            in summary.json so a run can be reproduced from it.
        evaluate_fn         evaluate(model, loader, device) -> (metrics, predictions)
        compute_metrics_fn  compute_metrics(labels, preds) -> dict, for the training side
        test_dataloader     when given, the best checkpoint is evaluated on it once,
                            after training. Pass None while selecting a model, so the
                            official test set is not used for selection.
        model_name          "rnn" or "transformer"; goes into the DataFrame and the
                            file names
        out_dir, run_id     where the results are written and what they are called

    Returns
        history   a pandas DataFrame, one row per epoch
        summary   a dict with the config, parameter counts, the best epoch and, if a
                  test loader was given, the test metrics

    Files written to out_dir
        <run_id>_history.csv              the DataFrame
        <run_id>_summary.json             the summary
        <run_id>_best.pt                  state_dict of the best epoch
        <run_id>_best_predictions.csv     predictions from that same epoch
        <run_id>_test_predictions.csv     only when a test loader was given
    """
    missing = [k for k in ("learning_rate", "epochs") if k not in params]
    if missing:
        raise KeyError(
            f"params is missing {missing}. When searching over learning rates, set the "
            f"value for each run, e.g. cfg = dict(M3_PARAMS); cfg['learning_rate'] = 1e-3")

    learning_rate = float(params["learning_rate"])
    epochs = int(params["epochs"])
    grad_clip = params.get("grad_clip")

    # ---- device, criterion, optimiser ----
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    run_id = run_id or f"{model_name}_{time.strftime('%m%d_%H%M%S')}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for n, p in model.named_parameters()
                if "embedding" in n and "position" not in n)

    rows, best_acc, best_epoch = [], -float("inf"), 0
    run_start = time.time()                     

    if verbose:
        print(f"[{run_id}] {model_name} | params {n_params:,} "
              f"(non-embedding {n_params - n_emb:,}) | lr {learning_rate} "
              f"| epochs {epochs} | device {device}")

    # ----for epoch_num in range(epochs) ----
    for epoch_num in range(epochs):

        epoch_start_time = time.time()           

        model.train()
        total_acc_train = 0
        total_loss_train = 0
        n_train = 0
        train_preds, train_labels, grad_norms = [], [], []

        for batch in tqdm(train_dataloader, disable=not verbose,
                          desc=f"epoch {epoch_num + 1}/{epochs}", leave=False):

            input_1, input_2, mask_1, mask_2, train_label = _unpack(batch, device)

            output = _logits_of(model(input_1, input_2, mask_1, mask_2))

            batch_loss = criterion(output, train_label.long())
            # Weighting by the size of this batch, before dividing by the total at the
            # end of the epoch, keeps a short final batch from counting as much as a
            # full one.
            total_loss_train += batch_loss.item() * train_label.size(0)

            preds = output.argmax(dim=1)
            total_acc_train += (preds == train_label).sum().item()
            n_train += train_label.size(0)

            # The predictions computed during this step are kept so the training metrics
            # can be worked out at the end of the epoch. This is why evaluate() is never
            # called on train_loader: it would cost a second forward pass over the whole
            # training set.
            train_preds += preds.detach().cpu().tolist()
            train_labels += train_label.detach().cpu().tolist()

            model.zero_grad()
            batch_loss.backward()
            # clip_grad_norm_ returns the norm before clipping, so the norm is recorded
            # for every batch. A sudden spike is the signal for an over-large learning
            # rate. With grad_clip unset the norm is measured but nothing is clipped.
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip if grad_clip else float("inf"))))
            optimizer.step()

        train_loss = total_loss_train / n_train
        # compute_metrics(labels, preds) -- labels first. Using the same function for
        # every model and baseline keeps the metrics on one definition.
        tm = _metrics(compute_metrics_fn(train_labels, train_preds))
        train_acc = tm["accuracy"] if tm["accuracy"] is not None else total_acc_train / n_train
        train_seconds = round(time.time() - epoch_start_time, 2)

        # ---- validation block, replaced by the shared evaluate() ----
        # evaluate() handles model.eval() and torch.no_grad() itself and restores the
        # previous mode afterwards, so calling it inside the loop is safe.
        val_start = time.time()
        metrics, predictions = evaluate_fn(model, val_dataloader, device)
        vm = _metrics(metrics)
        val_seconds = round(time.time() - val_start, 2)

        # ---- keep the epoch with the best validation score ----
        is_best = vm["accuracy"] is not None and vm["accuracy"] > best_acc
        if is_best:
            best_acc, best_epoch = vm["accuracy"], epoch_num + 1
            torch.save(model.state_dict(), out / f"{run_id}_best.pt")
            _save_predictions(predictions, out / f"{run_id}_best_predictions", epoch_num + 1)

        rows.append({
            "run_id": run_id, "model": model_name, "epoch": epoch_num + 1,
            "train_loss": train_loss, "train_accuracy": train_acc,
            "train_precision": tm["precision"], "train_recall": tm["recall"],
            "train_f1": tm["f1"],
            "valid_loss": vm["loss"], "valid_accuracy": vm["accuracy"],
            "valid_precision": vm["precision"], "valid_recall": vm["recall"],
            "valid_f1": vm["f1"],
            # train minus valid accuracy: plot this column to see overfitting directly
            "acc_gap": (train_acc - vm["accuracy"]) if vm["accuracy"] is not None else None,
            # gradient norms before clipping; a spike means the learning rate is too large
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
            "train_seconds": train_seconds,
            "valid_seconds": val_seconds,
            "epoch_seconds": round(time.time() - epoch_start_time, 2),
            "is_best": bool(is_best),
        })

        # The print line 
        if verbose:
            print(f'Epochs: {epoch_num + 1} '
                  f'| Train Loss: {train_loss: .3f} '
                  f'| Train Accuracy: {train_acc: .3f} '
                  f'| Val Loss: {_f(vm["loss"])} '
                  f'| Val Accuracy: {_f(vm["accuracy"])} '
                  f'| {timeSince(run_start, (epoch_num + 1) / epochs)}'
                  f'{"  * best" if is_best else ""}')

    history = pd.DataFrame(rows)
    history.to_csv(out / f"{run_id}_history.csv", index=False)

    summary = {
        "run_id": run_id, "model": model_name,
        "config": dict(params),
        "n_params": n_params, "n_params_no_embedding": n_params - n_emb,
        "best_epoch": best_epoch, "best_valid_accuracy": best_acc if rows else None,
        "checkpoint": str(out / f"{run_id}_best.pt"),
        "torch": torch.__version__, "device": str(device),
    }

    # ---- a single pass over the test set, on the best checkpoint ----
    if test_dataloader is not None and best_epoch:
        model.load_state_dict(torch.load(out / f"{run_id}_best.pt", map_location=device))
        test_metrics, test_predictions = evaluate_fn(model, test_dataloader, device)
        summary["test"] = _metrics(test_metrics)
        _save_predictions(test_predictions, out / f"{run_id}_test_predictions", None)
        if verbose:
            print(f'Test Accuracy: {_f(summary["test"]["accuracy"])} '
                  f'| Test F1: {_f(summary["test"]["f1"])} '
                  f'(best epoch {best_epoch})')

    with open(out / f"{run_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return history, summary


def _f(v, nd=3):
    return "  -  " if v is None else f"{v: .{nd}f}"
