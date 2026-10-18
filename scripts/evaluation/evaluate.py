import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ================ Metric Computation ========================
def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values).astype(int)


def compute_metrics(labels, preds) -> dict:
    """
    Compute evaluation metrics from ground-truth labels and model predictions.

    This function is shared by all models and baselines to ensure
    consistent metric definitions.

    Args:
        labels: Ground-truth labels, represented as a list, NumPy array,
                or tensor. Values must be 0 or 1.
        preds: Predicted labels in the same format as labels.

        Both labels and predictions use the following convention:
            0 = solution 1
            1 = solution 2

    Returns:
        A dictionary containing:
            accuracy: The proportion of correctly classified examples.
            precision: Macro-averaged precision across both classes.
            recall: Macro-averaged recall across both classes.
            f1: Macro-averaged F1 score across both classes.

        Precision, recall, and F1 are computed by treating each class
        as the positive class in turn, then averaging the results.
        Macro-averaged metrics can reveal class-selection imbalances
        that may not be apparent from accuracy alone.
    """
    labels = _to_numpy(labels)
    preds = _to_numpy(preds)

    if labels.shape != preds.shape:
        raise ValueError(f"labels and preds must have the same shape, got {labels.shape} and {preds.shape}")
    if labels.size == 0:
        raise ValueError("labels and preds must not be empty")

    precisions, recalls, f1s = [], [], []

    for c in (0, 1):
        # Calculate true positives, false positives, and false negatives
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))

        # Use zero when a denominator is zero, such as when the model
        # never predicts a particular class
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


# ================ Model Evaluation ========================
@torch.no_grad()
def evaluate(model, data_loader, device="cpu"):
    """
    Evaluate a model on a validation or test dataset without updating
    its parameters.

    Args:
        model: A TransformerClassifier or RNNWithAttention instance.
               Both models' output formats are supported.
        data_loader: A validation or test DataLoader returned by
                     get_piqa_dataloaders().
        device: The device on which the model runs, such as "cpu" or "cuda".

    Returns:
        metrics: A dictionary containing loss, accuracy, precision,
                 recall, and F1 score.

        predictions: A DataFrame with one row per example and the following
                     columns:
            row_id:    The example's position in the dataset, corresponding
                       to its row index after reset_index().
            label:     The ground-truth label (0 = solution 1, 1 = solution 2).
            logit_1:   The model's score for solution 1.
            logit_2:   The model's score for solution 2.
            prob_sol2: The model's predicted probability that solution 2 is correct.
            pred:      The model's predicted choice.
            correct:   Whether the prediction matches the ground-truth label.

    Notes:
        1. The model is switched to evaluation mode to disable dropout,
           then restored to its original training/evaluation mode.
        2. row_id assumes that the DataLoader preserves dataset order.
           The validation and test DataLoaders use shuffle=False.
           Do not use the training DataLoader if stable row IDs are required.
    """
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

            # RNNWithAttention returns (logits, weights), whereas
            # TransformerClassifier returns logits directly
            logits = outputs[0] if isinstance(outputs, tuple) else outputs  # [B, 2]

            # Accumulate the loss over all examples to compute the
            # dataset-wide mean loss after evaluation
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


# ================ McNemar's Test ========================

def mcnemar_test(correct_a, correct_b) -> dict:
    """
    Compare two models' correctness on the same set of examples using
    McNemar's test with continuity correction.

    Args:
        correct_a: Boolean array indicating whether model A is correct.
        correct_b: Boolean array indicating whether model B is correct.

    Returns:
        A dictionary containing:
            only_a: Number of examples correctly classified only by model A.
            only_b: Number of examples correctly classified only by model B.
            chi2: Continuity-corrected McNemar test statistic.
            p_value: Corresponding asymptotic p-value.

    If both models agree on every example's correctness, there are no
    discordant pairs and the p-value is set to 1.
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
