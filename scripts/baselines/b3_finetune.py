"""B3: Fine-tune pretrained encoders for binary choice classification.

distilbert-base-uncased has approximately 66 million parameters, while
roberta-base has approximately 125 million parameters.
Our models trained from scratch have approximately 1 million parameters,
making them about 60 to 125 times smaller.
Both models are pretrained, so this comparison concerns training strategies
rather than models of equivalent size.
The pretraining corpora may contain PIQA-related text; this must be
acknowledged as a limitation in the report.

Use each model's own tokenizer. Do not pass piqa_bpe_v3 token IDs to these models.
Each question is represented by two sentence pairs: (goal, sol1) and (goal, sol2).

Run from the repository root:

    python -m scripts.baselines.b3_finetune

Set the random seed before loading the model. Keep the parameters from the
epoch with the highest validation accuracy in memory, then restore them after
training. Evaluate the test set only once using the restored model.
Do not save model weights.
If a seed fails, record the failure without rerunning it or selecting results.
"""

import argparse
import copy
import inspect
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.baselines.results import (
    library_versions,
    prediction_frame,
    softmax_sol2,
    write_json,
)
from scripts.data_loader.load_piqa_data import piqa_to_dataframe
from scripts.evaluation.evaluate import compute_metrics
from scripts.training.trainer import set_seed

DATA_DIR = Path("datasets/PIQA")
OUTPUT_DIR = Path("reports/baselines/b3_finetune")
DEFAULT_MODELS = ("distilbert-base-uncased", "roberta-base")
DEFAULT_SEEDS = (4012, 4013, 4014)


class PIQAChoiceDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return str(row["goal"]), str(row["sol1"]), str(row["sol2"]), int(row["label"])


def collate_choices(batch, tokenizer, max_length):
    """Dynamically pad both choices into [B, 2, L], where L is the longest
    sentence pair in the batch and does not exceed max_length.
    """
    goals, sol1, sol2, labels = zip(*batch)
    first, second = [], []
    for goal, left, right in zip(goals, sol1, sol2):
        first.extend((goal, goal))
        second.extend((left, right))
    encoded = tokenizer(
        first,
        second,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_size = len(batch)
    model_inputs = {
        "input_ids": encoded["input_ids"].contiguous().view(batch_size, 2, -1),
        "attention_mask": encoded["attention_mask"].contiguous().view(batch_size, 2, -1),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if "token_type_ids" in encoded:
        model_inputs["token_type_ids"] = encoded["token_type_ids"].contiguous().view(batch_size, 2, -1)
    return model_inputs


def truncation_rate(frame, tokenizer, max_length):
    """Calculate the proportion of (goal, solution) pairs exceeding max_length."""
    if len(frame) == 0:
        return {"n_choices": 0, "n_truncated": 0, "rate": 0.0}
    goals = frame["goal"].astype(str).tolist()
    # Only measure sequence lengths. Temporarily increase model_max_length to
    # prevent warnings for long sequences that are intentionally not truncated.
    previous_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e9)
    try:
        encoded = tokenizer(
            goals + goals,
            frame["sol1"].astype(str).tolist() + frame["sol2"].astype(str).tolist(),
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
    finally:
        tokenizer.model_max_length = previous_max
    lengths = [len(ids) for ids in encoded["input_ids"]]
    n_truncated = sum(length > max_length for length in lengths)
    return {
        "n_choices": len(lengths),
        "n_truncated": int(n_truncated),
        "rate": float(n_truncated / len(lengths)),
    }


def _move(batch, device, use_token_type_ids):
    moved = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    if use_token_type_ids and "token_type_ids" in batch:
        moved["token_type_ids"] = batch["token_type_ids"].to(device)
    if "labels" in batch:
        moved["labels"] = batch["labels"].to(device)
    return moved


def _loader(frame, tokenizer, batch_size, max_length, shuffle, seed):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        PIQAChoiceDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: collate_choices(batch, tokenizer, max_length),
    )


def _lr_at(step, total_steps, warmup_steps):
    """Return the learning-rate multiplier for a zero-indexed step.

    Increase linearly to 1 during warmup_steps, then decrease linearly to 0.
    """
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    remain = total_steps - warmup_steps
    return max(0.0, float(total_steps - step) / float(max(1, remain)))


@torch.no_grad()
def predict_frame(model, frame, tokenizer, batch_size, max_length, device, use_token_type_ids):
    """Evaluate the model in eval mode using float32 without automatic mixed precision."""
    was_training = model.training
    model.eval()
    loader = _loader(frame, tokenizer, batch_size, max_length, shuffle=False, seed=0)
    all_logits = []
    all_labels = []
    total_loss = 0.0
    try:
        for batch in loader:
            labels = batch["labels"]
            moved = _move(batch, device, use_token_type_ids)
            output = model(**moved)
            logits = output.logits.float()
            total_loss += torch.nn.functional.cross_entropy(
                logits, moved["labels"], reduction="sum").item()
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    finally:
        model.train(was_training)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    predictions = prediction_frame(
        labels,
        logits[:, 0].numpy(),
        logits[:, 1].numpy(),
        softmax_sol2(logits[:, 0].numpy(), logits[:, 1].numpy()),
        logits.argmax(dim=1).numpy(),
    )
    metrics = compute_metrics(predictions["label"], predictions["pred"])
    metrics["loss"] = float(total_loss / len(labels))
    metrics["n"] = int(len(labels))
    return metrics, predictions


def load_multiple_choice(model_name):
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name, dtype=torch.float32)
    accepts_token_types = "token_type_ids" in inspect.signature(model.forward).parameters
    return model, tokenizer, accepts_token_types


def train_one_seed(
    model_name, seed, train_df, valid_df, test_df, epochs, lr, batch_size,
    max_length, splits, output_dir,
):
    started = time.perf_counter()
    # Multithreaded matrix operations on CPU may use different addition orders,
    # making repeated runs with the same seed inconsistent.
    torch.set_num_threads(1)
    set_seed(seed)
    model, tokenizer, use_token_type_ids = load_multiple_choice(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    use_amp = device.type == "cuda"
    print(
        f"{model_name} seed {seed} params={n_params} device={device} "
        f"train_amp={'fp16' if use_amp else 'off'} eval_dtype=float32"
    )

    truncation = {
        "max_length": max_length,
        "train": truncation_rate(train_df, tokenizer, max_length),
        "valid": truncation_rate(valid_df, tokenizer, max_length),
    }
    if test_df is not None:
        truncation["test"] = truncation_rate(test_df, tokenizer, max_length)
    for split_name, report in truncation.items():
        if split_name == "max_length":
            continue
        print(
            f"  truncation {split_name}: {report['n_truncated']}/{report['n_choices']} "
            f"= {report['rate']:.4f}"
        )

    train_loader = _loader(train_df, tokenizer, batch_size, max_length, shuffle=True, seed=seed)
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(total_steps * 0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_accuracy = -1.0
    best_epoch = None
    best_state = None
    history_rows = []

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for step, batch in enumerate(train_loader):
            moved = _move(batch, device, use_token_type_ids)
            optimizer.zero_grad(set_to_none=True)
            global_step = (epoch - 1) * len(train_loader) + step
            for group in optimizer.param_groups:
                group["lr"] = lr * _lr_at(global_step, total_steps, warmup_steps)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**moved)
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_n = int(moved["labels"].size(0))
            running_loss += float(loss.detach().float().item()) * batch_n
            preds = output.logits.float().argmax(dim=1)
            correct += int((preds == moved["labels"]).sum().item())
            seen += batch_n

        valid_metrics, _valid_predictions = predict_frame(
            model, valid_df, tokenizer, batch_size, max_length, device, use_token_type_ids)
        is_best = valid_metrics["accuracy"] > best_accuracy
        if is_best:
            best_accuracy = valid_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history_rows.append({
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy": correct / seen,
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_precision": valid_metrics["precision"],
            "valid_recall": valid_metrics["recall"],
            "valid_f1": valid_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "is_best": bool(is_best),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
        })
        print(
            f"  epoch {epoch} train_acc {correct / seen:.4f} "
            f"valid_acc {valid_metrics['accuracy']:.4f}"
            f"{'  best' if is_best else ''}"
        )

    if best_state is None:
        raise RuntimeError("no epoch was evaluated")
    model.load_state_dict(best_state)

    seed_dir = Path(output_dir) / alias_of(model_name) / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(history_rows)
    history.to_csv(seed_dir / "history.csv", index=False)

    valid_metrics, valid_predictions = predict_frame(
        model, valid_df, tokenizer, batch_size, max_length, device, use_token_type_ids)
    valid_predictions.to_csv(seed_dir / "valid_predictions.csv", index=False)

    test_metrics = None
    if "test" in splits and test_df is not None:
        test_metrics, test_predictions = predict_frame(
            model, test_df, tokenizer, batch_size, max_length, device, use_token_type_ids)
        test_predictions.to_csv(seed_dir / "test_predictions.csv", index=False)
        print(f"  test accuracy {test_metrics['accuracy']:.4f} (best epoch {best_epoch})")

    import transformers
    payload = {
        "baseline": "b3_finetune",
        "model_name": model_name,
        "seed": int(seed),
        "n_params": int(n_params),
        "best_epoch": int(best_epoch),
        "selection": {"split": "valid", "metric": "accuracy", "best_epoch": int(best_epoch)},
        "valid": valid_metrics,
        "test": test_metrics,
        "truncation": truncation,
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "weight_decay": 0.01,
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "grad_clip": 1.0,
        "warmup_steps": int(warmup_steps),
        "total_steps": int(total_steps),
        "train_amp": "fp16" if use_amp else "off",
        "eval_dtype": "float32",
        "n_train": int(len(train_df)),
        "seconds": round(time.perf_counter() - started, 3),
        "note": "The model is approximately 60 to 125 times larger than models trained from scratch and has been pretrained. The pretraining corpus may contain PIQA-related text.",
        "versions": library_versions({"transformers": transformers.__version__}),
    }
    write_json(seed_dir / "metrics.json", payload)
    print(f"wrote {seed_dir}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _summarize(model_name, results, failures):
    def column(split, key):
        values = [
            result[split][key]
            for result in results
            if result.get(split)
        ]
        if not values:
            return None, None
        array = np.asarray(values, dtype=np.float64)
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        return float(array.mean()), std

    valid_mean, valid_std = column("valid", "accuracy")
    test_mean, test_std = column("test", "accuracy")
    return {
        "baseline": "b3_finetune",
        "model_name": model_name,
        "seeds": [result["seed"] for result in results],
        "failed_seeds": failures,
        "valid_accuracy_mean": valid_mean,
        "valid_accuracy_std": valid_std,
        "test_accuracy_mean": test_mean,
        "test_accuracy_std": test_std,
        "runs": [
            {
                "seed": result["seed"],
                "best_epoch": result["best_epoch"],
                "valid_accuracy": result["valid"]["accuracy"],
                "test_accuracy": None if result["test"] is None else result["test"]["accuracy"],
            }
            for result in results
        ],
    }


def alias_of(model_name):
    return model_name.split("/")[-1].lower()


def take(frame, limit):
    frame = frame.reset_index(drop=True)
    if limit is None:
        return frame
    return frame.iloc[:limit].reset_index(drop=True)


def run(
    models, seeds, epochs, lr, batch_size, max_length, limit_train, limit_eval,
    splits, output_dir, data_dir=DATA_DIR,
):
    _train_df, valid_df, test_df = piqa_to_dataframe(data_dir)
    train_df = take(_train_df, limit_train)
    valid_df = take(valid_df, limit_eval)
    test_df = take(test_df, limit_eval) if "test" in splits else None
    output_dir = Path(output_dir)
    summaries = []
    for model_name in models:
        results = []
        failures = []
        for seed in seeds:
            try:
                results.append(train_one_seed(
                    model_name, int(seed), train_df, valid_df, test_df,
                    epochs, lr, batch_size, max_length, splits, output_dir,
                ))
            except Exception as exc:
                failure = {"seed": int(seed), "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(f"FAILED {model_name} seed {seed}: {failure['error']}")
                traceback.print_exc()
        summary = _summarize(model_name, results, failures)
        summary_path = output_dir / alias_of(model_name) / "summary.json"
        write_json(summary_path, summary)
        print(f"wrote {summary_path}")
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        splits=set(args.splits),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
