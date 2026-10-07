import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ================ 指标计算 ========================
def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values).astype(int)


def compute_metrics(labels, preds) -> dict:
    '''
    根据正确答案和模型的选择计算指标。所有模型和 baseline 都用这个函数，保证指标口径一致。

    labels, preds: 等长的 0/1 序列（list / numpy / tensor 都可以），0 = sol1，1 = sol2

    返回:
        accuracy:  选对的比例
        precision / recall / f1: 把 "sol1 正确" 和 "sol2 正确" 各当一次正类分别计算，再取平均（macro average）
                                 如果模型总偏向选某一边，accuracy 看不出来，但 f1 会明显下降
    '''
    labels = _to_numpy(labels)
    preds = _to_numpy(preds)

    if labels.shape != preds.shape:
        raise ValueError(f"labels and preds must have the same shape, got {labels.shape} and {preds.shape}")
    if labels.size == 0:
        raise ValueError("labels and preds must not be empty")

    precisions, recalls, f1s = [], [], []

    for c in (0, 1):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))

        # 分母为 0 时（比如模型从来不选这一类）按 0 处理
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": float(np.mean(preds == labels)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


# ================ 模型评估 ========================
@torch.no_grad()
def evaluate(model, data_loader, device="cpu"):
    '''
    在验证集或测试集上评估模型：只做 forward，不更新参数。

    参数:
        model:       TransformerClassifier 或 RNNWithAttention（两种 forward 返回格式都兼容）
        data_loader: get_piqa_dataloaders() 返回的 validate_loader 或 test_loader
        device:      "cpu" 或 "cuda"，需要和 model 所在的设备一致

    返回:
        metrics:     dict，包含 loss / accuracy / precision / recall / f1
        predictions: DataFrame，每道题一行
            row_id     题目在数据集里的位置，对应 piqa_to_dataframe() 返回的 DataFrame 在 reset_index 之后的行号
            label      正确答案（0 = sol1，1 = sol2）
            logit_1    模型给 sol1 的分数
            logit_2    模型给 sol2 的分数
            prob_sol2  模型认为 sol2 正确的概率
            pred       模型的选择
            correct    是否选对

    注意:
        1. 函数内部会切换到 model.eval()（关闭 dropout），结束后恢复调用前的模式，在训练循环里调用也不影响训练
        2. row_id 依赖 DataLoader 不打乱顺序，validate_loader 和 test_loader 都是 shuffle=False，不要传入 train_loader
    '''
    was_training = model.training
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []

    try:
        for input_1, input_2, mask_1, mask_2, labels in data_loader:
            input_1, input_2 = input_1.to(device), input_2.to(device)
            mask_1, mask_2 = mask_1.to(device), mask_2.to(device)
            labels = labels.to(device)

            outputs = model(input_1, input_2, mask_1, mask_2)

            # RNNWithAttention 返回 (logits, weights)，TransformerClassifier 只返回 logits
            logits = outputs[0] if isinstance(outputs, tuple) else outputs  # [B, 2]

            # 先累加每道题的 loss，最后除以总题数，得到整个数据集的平均 loss
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    finally:
        model.train(was_training)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    predictions = pd.DataFrame({
        "row_id": np.arange(len(labels)),
        "label": labels.numpy(),
        "logit_1": logits[:, 0].numpy(),
        "logit_2": logits[:, 1].numpy(),
        "prob_sol2": probs[:, 1].numpy(),
        "pred": preds.numpy(),
    })
    predictions["correct"] = predictions["pred"] == predictions["label"]

    metrics = {"loss": total_loss / len(labels)}
    metrics.update(compute_metrics(labels, preds))

    return metrics, predictions


def mcnemar_test(correct_a, correct_b) -> dict:
    """配对比较两个模型在同一批题上的对错。

    only_a / only_b 是只有一方做对的题数。chi2 使用连续性校正
    (|b - c| - 1)^2 / (b + c)。b + c = 0 时没有分歧，p 取 1。
    """
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError(
            f"correct_a and correct_b must have the same shape, "
            f"got {correct_a.shape} and {correct_b.shape}")

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b
    if discordant == 0:
        return {"only_a": only_a, "only_b": only_b, "chi2": 0.0, "p_value": 1.0}

    chi2 = (abs(only_a - only_b) - 1) ** 2 / discordant
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "only_a": only_a,
        "only_b": only_b,
        "chi2": float(chi2),
        "p_value": float(p_value),
    }
