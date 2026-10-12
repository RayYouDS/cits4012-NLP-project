"""基线共用的预测表和 metrics.json。列和 evaluate() 的 predictions 一致。"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


COLUMNS = ["row_id", "label", "logit_1", "logit_2", "prob_sol2", "pred", "correct"]


def library_versions(extra=None):
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if extra:
        versions.update(extra)
    return versions


def prediction_frame(labels, logit_1, logit_2, prob_sol2, pred):
    """row_id 是 reset_index(drop=True) 之后的行号，用来和主模型逐题对齐。"""
    labels = np.asarray(labels)
    pred = np.asarray(pred).astype(int)
    frame = pd.DataFrame({
        "row_id": np.arange(len(labels)),
        "label": labels.astype(int),
        "logit_1": np.asarray(logit_1, dtype=float),
        "logit_2": np.asarray(logit_2, dtype=float),
        "prob_sol2": np.asarray(prob_sol2, dtype=float),
        "pred": pred,
    })
    frame["correct"] = frame["pred"].to_numpy() == frame["label"].to_numpy()
    return frame


def softmax_sol2(logit_1, logit_2):
    logit_1 = np.asarray(logit_1, dtype=np.float64)
    logit_2 = np.asarray(logit_2, dtype=np.float64)
    shift = np.maximum(logit_1, logit_2)
    exp_1 = np.exp(logit_1 - shift)
    exp_2 = np.exp(logit_2 - shift)
    return exp_2 / (exp_1 + exp_2)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
