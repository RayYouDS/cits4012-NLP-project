"""M3：GRU、LSTM、Transformer 在若干学习率和种子上训练，只看验证集。

从仓库根目录运行：

    python -m scripts.training.run_m3 --config config/m3_params.json

不重新训练 tokenizer，也不改数据切分。tokenize_piqa_dataframe 返回什么划分，这里就用什么划分。
不使用测试集。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from scripts.data_loader.piqa_dataloader import (
    PIQACollator,
    PIQADataset,
    tokenize_piqa_dataframe,
)
from scripts.evaluation.evaluate import evaluate, mcnemar_test
from scripts.models.rnn_attention_refactored import RNNWithAttention
from scripts.models.transformer_bi_classifier import TransformerClassifier
from scripts.training.bucket_sampler import LengthBucketBatchSampler, piqa_lengths
from scripts.training.trainer import set_seed, train

DATA_DIR = Path("datasets/PIQA")
ALPHA = 0.05


def build_model(name, vocab_size, pad_id, cfg):
    if name in {"rnn", "gru", "lstm"}:
        return RNNWithAttention(
            vocab_size=vocab_size,
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_size"],
            padding_idx=pad_id,
            architecture=name,
            bidirectional=cfg["bidirectional"],
            enable_attention=cfg["enable_attention"],
        )
    if name == "transformer":
        return TransformerClassifier(
            vocab_size=vocab_size,
            embedding_dim=cfg["embedding_dim"],
            padding_idx=pad_id,
            max_seq_length=2048,
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
        )
    raise ValueError(f"unknown model {name}")


def build_optimizer(name, parameters, lr, weight_decay=None, momentum=0.9):
    """weight_decay 为 None 时不传入，沿用该优化器的 PyTorch 默认值。momentum 只用于 SGD。"""
    name = name.lower()
    kwargs = {"lr": float(lr)}
    if weight_decay is not None:
        kwargs["weight_decay"] = float(weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if name == "sgd":
        kwargs["momentum"] = 0.9 if momentum is None else float(momentum)
        return torch.optim.SGD(parameters, **kwargs)
    raise ValueError(f"unknown optimizer {name}")


def make_loader(df, collator, batch_size, num_workers, pin_memory, shuffle, generator=None, lengths=None):
    dataset = PIQADataset(df)
    kwargs = {
        "collate_fn": collator,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": pin_memory,
    }
    if lengths is None:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, **kwargs)
    sampler = LengthBucketBatchSampler(lengths, batch_size, generator=generator)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)


def _as_bool(series):
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(["true", "1"]).to_numpy()


def select_table(summary):
    """每个模型用各 seed 的平均验证准确率选出学习率。"""
    rows = []
    for model, group in summary.groupby("model", sort=False):
        stats = group.groupby("learning_rate")["valid_accuracy"].agg(["mean", "std", "count"])
        learning_rate = stats["mean"].idxmax()
        n_seeds = int(stats.loc[learning_rate, "count"])
        std = stats.loc[learning_rate, "std"]
        rows.append({
            "model": model,
            "learning_rate": float(learning_rate),
            "valid_accuracy_mean": float(stats.loc[learning_rate, "mean"]),
            "valid_accuracy_std": 0.0 if n_seeds < 2 or pd.isna(std) else float(std),
            "n_seeds": n_seeds,
        })
    table = pd.DataFrame(rows)
    return table.sort_values(
        ["valid_accuracy_mean", "model"], ascending=[False, True]
    ).reset_index(drop=True)


def plot_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run_id, history in histories:
        axes[0].plot(history["epoch"], history["valid_accuracy"], label=run_id)
        gap = history["train_eval_accuracy"] - history["valid_accuracy"]
        axes[1].plot(history["epoch"], gap, label=run_id)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("valid accuracy")
    axes[0].set_title("validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("train_eval - valid")
    axes[1].set_title("generalisation gap in eval mode")
    for axis in axes:
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg):
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    num_workers = int(cfg["num_workers"])

    tokenizer = spm.SentencePieceProcessor(model_file=cfg["tokenizer"])
    train_df, valid_df, _test_df = tokenize_piqa_dataframe(DATA_DIR, tokenizer)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    train_eval_df = train_df.sample(n=len(valid_df), random_state=4012).reset_index(drop=True)

    collator = PIQACollator(tokenizer)
    lengths = piqa_lengths(train_df)
    vocab_size = tokenizer.get_piece_size()
    pad_id = tokenizer.pad_id()

    print(
        f"device={device} | train {len(train_df)} | valid {len(valid_df)} "
        f"| vocab {vocab_size} | workers {num_workers}"
    )
    print("test set is not loaded into a DataLoader")

    val_loader = make_loader(
        valid_df, collator, cfg["batch_size"], num_workers, pin_memory, shuffle=False)
    train_eval_loader = make_loader(
        train_eval_df, collator, cfg["batch_size"], num_workers, pin_memory, shuffle=False)

    summary_rows = []
    histories = []

    for seed in cfg["seeds"]:
        for model_name in cfg["models"]:
            for lr in cfg["learning_rates"]:
                lr = float(lr)
                seed = int(seed)
                run_id = f"m3_{model_name}_lr{lr:g}_seed{seed}"
                print("=" * 72)
                print(run_id)

                generator = torch.Generator()
                generator.manual_seed(seed)
                train_loader = make_loader(
                    train_df, collator, cfg["batch_size"], num_workers, pin_memory,
                    shuffle=True, generator=generator, lengths=lengths,
                )

                set_seed(seed)
                model = build_model(model_name, vocab_size, pad_id, cfg).to(device)
                n_params = sum(p.numel() for p in model.parameters())
                optimizer = build_optimizer(
                    cfg["optimizer"], model.parameters(), lr,
                    weight_decay=cfg.get("weight_decay"),
                    momentum=cfg.get("momentum", 0.9),
                )

                model, history = train(
                    model, train_loader, val_loader, optimizer, device,
                    max_epochs=int(cfg["max_epochs"]),
                    patience=cfg.get("patience"),
                    grad_clip=cfg.get("grad_clip"),
                    train_eval_loader=train_eval_loader,
                    verbose=True,
                )

                torch.save(model.state_dict(), output_dir / f"{run_id}_best.pt")
                history.to_csv(output_dir / f"{run_id}_history.csv", index=False)
                val_metrics, predictions = evaluate(model, val_loader, device)
                predictions.to_csv(output_dir / f"{run_id}_val_predictions.csv", index=False)

                best = history.loc[history["valid_accuracy"].idxmax()]
                summary_rows.append({
                    "model": model_name,
                    "learning_rate": lr,
                    "seed": seed,
                    "best_epoch": int(best["epoch"]),
                    "epochs_run": int(len(history)),
                    "valid_accuracy": float(val_metrics["accuracy"]),
                    "valid_f1": float(val_metrics["f1"]),
                    "train_eval_accuracy": float(best["train_eval_accuracy"]),
                    "gap_at_best": float(best["train_eval_accuracy"] - best["valid_accuracy"]),
                    "seconds_to_best": float(history.loc[history["epoch"] <= best["epoch"], "epoch_seconds"].sum()),
                    "seconds_total": float(history["epoch_seconds"].sum()),
                    "params": int(n_params),
                    "run_id": run_id,
                })
                histories.append((run_id, history))

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "m3_summary.csv"
    summary.drop(columns=["run_id"]).to_csv(summary_path, index=False)

    table = select_table(summary)
    table.to_csv(output_dir / "m3_model_table.csv", index=False)
    print("\nmodel table")
    print(table.to_string(index=False))

    plot_curves(histories, output_dir / "m3_curves.png")

    selection = {"alpha": ALPHA, "ranking": table.to_dict(orient="records")}
    if len(table) >= 2:
        first, second = table.iloc[0], table.iloc[1]
        seed0 = int(cfg["seeds"][0])
        def predictions_of(row):
            run_id = f"m3_{row['model']}_lr{float(row['learning_rate']):g}_seed{seed0}"
            return pd.read_csv(output_dir / f"{run_id}_val_predictions.csv")

        pred_a = predictions_of(first)
        pred_b = predictions_of(second)
        merged = pred_a.merge(pred_b, on="row_id", suffixes=("_a", "_b"))
        compared = mcnemar_test(_as_bool(merged["correct_a"]), _as_bool(merged["correct_b"]))
        significant = compared["p_value"] < ALPHA
        selection.update({
            "seed": seed0,
            "first": {"model": first["model"], "learning_rate": float(first["learning_rate"])},
            "second": {"model": second["model"], "learning_rate": float(second["learning_rate"])},
            "mcnemar": compared,
            "significant": significant,
        })
        verdict = "显著" if significant else "不显著"
        print(
            f"\nMcNemar {first['model']} vs {second['model']} "
            f"(seed {seed0}): only_a={compared['only_a']} only_b={compared['only_b']} "
            f"chi2={compared['chi2']:.4f} p={compared['p_value']:.4f} "
            f"差别{verdict} (alpha={ALPHA})"
        )
    else:
        print("\nfewer than 2 models, skip McNemar")

    with open(output_dir / "m3_selection.json", "w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
    print(f"wrote {output_dir}")
    return summary, table, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m3_params.json")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    run(cfg)


if __name__ == "__main__":
    main()
