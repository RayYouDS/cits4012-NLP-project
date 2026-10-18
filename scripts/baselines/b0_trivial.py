"""B0: Non-trained rule-based baselines.

Run from the repository root:

    python -m scripts.baselines.b0_trivial

All three rules are fixed. Both validation and test sets are evaluated, and
no model selection is performed on the validation set.
The random baseline uses NumPy seed 4012 to generate predictions once:
the validation set is processed first, followed by the test set, using the
same Generator instance.
"""

import time
from pathlib import Path

import numpy as np

from scripts.baselines.results import library_versions, prediction_frame, write_json
from scripts.data_loader.load_piqa_data import piqa_to_dataframe
from scripts.evaluation.evaluate import compute_metrics

DATA_DIR = Path("datasets/PIQA")
OUTPUT_DIR = Path("reports/baselines/b0_trivial")
SEED = 4012


def _rule_frame(df, pred, prob_sol2):
    pred = np.asarray(pred).astype(int)
    logit_1 = (pred == 0).astype(float)
    logit_2 = (pred == 1).astype(float)
    if np.ndim(prob_sol2) == 0:
        prob_sol2 = np.full(len(pred), float(prob_sol2))
    return prediction_frame(df["label"], logit_1, logit_2, prob_sol2, pred)


def random_pred(n, rng):
    return rng.integers(0, 2, size=n)


def always_sol1_pred(n):
    return np.zeros(n, dtype=int)


def shorter_pred(df):
    """Choose the shorter solution by character count; choose sol1 if lengths are equal."""
    len_1 = df["sol1"].astype(str).str.len().to_numpy()
    len_2 = df["sol2"].astype(str).str.len().to_numpy()
    return (len_2 < len_1).astype(int)


def _metrics(frame):
    metrics = compute_metrics(frame["label"], frame["pred"])
    metrics["n"] = int(len(frame))
    return metrics


def run(output_dir=OUTPUT_DIR, data_dir=DATA_DIR):
    started = time.perf_counter()
    _train_df, valid_df, test_df = piqa_to_dataframe(data_dir)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    valid_short = shorter_pred(valid_df)
    test_short = shorter_pred(test_df)
    rules = {
        "random": (
            _rule_frame(valid_df, random_pred(len(valid_df), rng), 0.5),
            _rule_frame(test_df, random_pred(len(test_df), rng), 0.5),
        ),
        "always_sol1": (
            _rule_frame(valid_df, always_sol1_pred(len(valid_df)), 0.0),
            _rule_frame(test_df, always_sol1_pred(len(test_df)), 0.0),
        ),
        "shorter": (
            _rule_frame(valid_df, valid_short, valid_short.astype(float)),
            _rule_frame(test_df, test_short, test_short.astype(float)),
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (valid_frame, test_frame) in rules.items():
        valid_frame.to_csv(output_dir / f"{name}_valid_predictions.csv", index=False)
        test_frame.to_csv(output_dir / f"{name}_test_predictions.csv", index=False)
        report[name] = {"valid": _metrics(valid_frame), "test": _metrics(test_frame)}
        print(
            f"{name:12} valid {report[name]['valid']['accuracy']:.4f}  "
            f"test {report[name]['test']['accuracy']:.4f}"
        )

    label_rate = {
        "valid": float((valid_df["label"] == 0).mean()),
        "test": float((test_df["label"] == 0).mean()),
    }
    payload = {
        "baseline": "b0_trivial",
        "model_name": "rules",
        "n_params": 0,
        "selection": None,
        "note": "Rules are fixed. The random baseline uses NumPy seed 4012, processing the validation set before the test set. The shorter baseline chooses sol1 when both solutions have equal lengths.",
        "label_0_rate": label_rate,
        "rules": report,
        "seconds": round(time.perf_counter() - started, 3),
        "versions": library_versions(),
    }
    write_json(output_dir / "metrics.json", payload)
    print(f"wrote {output_dir}")
    return payload


def main():
    run()


if __name__ == "__main__":
    main()
