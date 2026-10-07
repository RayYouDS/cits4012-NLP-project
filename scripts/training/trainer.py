"""PIQA 训练循环。

优化器和存盘由调用方负责。损失函数固定为 CrossEntropyLoss。
train() 只更新传入的模型，验证准确率最好的那一轮参数留在内存里，结束时载回。
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
    """固定 random、numpy、torch、cuda 的种子。在创建模型之前调用。

    训练集每个 epoch 的 batch 顺序要另外用 DataLoader / BatchSampler 自己的
    torch.Generator 来固定，由调用方 manual_seed 之后传进去。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _logits_of(output):
    """RNNWithAttention 返回 (logits, weights)，TransformerClassifier 只返回 logits。"""
    return output[0] if isinstance(output, (tuple, list)) else output


def _to(batch, device):
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
    """训练最多 max_epochs 轮，每轮用 evaluate() 看一次验证集。

    损失函数固定为 nn.CrossEntropyLoss()（mean reduction）。loss 先乘 batch 大小，
    再在整轮末除以题数，和 evaluate() 的「总 loss / 题数」同一口径。

    patience 传入整数时，验证准确率连续这么多轮没有刷新最好成绩就停止。
    patience 为 None 时跑满 max_epochs。无论是否早停，返回的 model 都已载入
    验证准确率最高那一轮的参数。本函数不写任何文件。
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

        # 一整轮只同步一次，避免每个 batch 都把 GPU 卡住等 CPU。
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
