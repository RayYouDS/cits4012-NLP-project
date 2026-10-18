"""B2: Zero-shot language model scoring.

These models are approximately 100 to 500 times larger than our models
trained from scratch. The comparison focuses on training approaches
(training from scratch versus no task-specific training and relying solely
on pretraining), rather than models of comparable size.
The pretraining corpora may already contain PIQA data, creating a risk of
data contamination. This must be acknowledged as a limitation in the report.

Each model must use its own tokenizer. Token IDs from piqa_bpe_v3 must not
be passed to these models.

Run from the repository root:

    python -m scripts.baselines.b2_zero_shot_lm

Compare sum and mean scoring on the validation set only. After selecting
one scoring method, evaluate only that method on the test set.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from scripts.baselines.results import (
    library_versions,
    prediction_frame,
    softmax_sol2,
    write_json,
)
from scripts.data_loader.load_piqa_data import piqa_to_dataframe
from scripts.evaluation.evaluate import compute_metrics

DATA_DIR = Path("datasets/PIQA")
OUTPUT_DIR = Path("reports/baselines/b2_zero_shot_lm")
DEFAULT_MODELS = ("openai-community/gpt2", "Qwen/Qwen2.5-0.5B")


def format_context(goal):
    return f"Question: {goal}\nAnswer:"


def format_continuation(solution):
    return " " + solution


def encode_pair(tokenizer, context, continuation):
    """Encode the context and continuation separately, then concatenate them.

    Only the continuation tokens are included in the scoring.
    """
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if len(context_ids) == 0:
        raise ValueError("context produced no tokens")
    if len(continuation_ids) == 0:
        raise ValueError("continuation produced no tokens")
    return context_ids, continuation_ids


def _continuation_sum(log_probs, input_ids, context_len, continuation_len):
    """Sum the log probabilities of the continuation tokens.

    logits[t] predicts the token at position t + 1. Therefore, the first
    continuation token is predicted by the final context position.
    """
    token_ids = input_ids[context_len:context_len + continuation_len]
    chosen = log_probs[context_len - 1:context_len + continuation_len - 1]
    token_logprob = chosen.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return float(token_logprob.sum().item()), continuation_len


def _forward_log_probs(model, input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return torch.log_softmax(output.logits.float(), dim=-1)


@torch.no_grad()
def sum_logprobs(model, pairs, batch_size, device):
    """Score each continuation by summing its token log probabilities.

    Args:
        pairs: A sequence of (context_ids, continuation_ids) pairs.

    Returns:
        A tuple containing the sum of continuation token log probabilities
        and the corresponding number of continuation tokens for each pair.

    On the GPU, sequences are right-padded within each batch, and the
    attention mask marks padding tokens with 0. Real tokens remain at
    positions 0 through L - 1.

    On the CPU, float32 matrix multiplication can vary with batch shape.
    The sum for the same sequence can differ by approximately 3e-4 between
    batch sizes of 1 and 8. Therefore, CPU inference processes sequences
    individually so that the scores remain consistent across batch sizes.
    """
    totals = np.empty(len(pairs), dtype=np.float64)
    counts = np.empty(len(pairs), dtype=np.int64)
    if not pairs:
        return totals, counts
    if device.type == "cpu":
        return _sum_logprobs_single(model, pairs, totals, counts, device)
    return _sum_logprobs_padded(model, pairs, batch_size, totals, counts, device)


def _sum_logprobs_single(model, pairs, totals, counts, device):
    for index, (context_ids, continuation_ids) in enumerate(pairs):
        sequence = context_ids + continuation_ids
        input_ids = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        log_probs = _forward_log_probs(model, input_ids, attention_mask)
        totals[index], counts[index] = _continuation_sum(
            log_probs[0], input_ids[0], len(context_ids), len(continuation_ids))
        done = index + 1
        if done == 1 or done == len(pairs) or done % 50 == 0:
            print(f"  scored {done}/{len(pairs)}")
    return totals, counts


def _sum_logprobs_padded(model, pairs, batch_size, totals, counts, device):
    if model.config.pad_token_id is None:
        raise ValueError("model.config.pad_token_id is not set")
    pad_id = int(model.config.pad_token_id)

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        sequences = [context_ids + continuation_ids for context_ids, continuation_ids in chunk]
        width = max(len(sequence) for sequence in sequences)
        batch = len(chunk)
        input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :len(sequence)] = 1

        log_probs = _forward_log_probs(model, input_ids.to(device), attention_mask.to(device))
        for row, (context_ids, continuation_ids) in enumerate(chunk):
            totals[start + row], counts[start + row] = _continuation_sum(
                log_probs[row], input_ids[row].to(device), len(context_ids), len(continuation_ids))

        done = min(start + batch_size, len(pairs))
        if start == 0 or done == len(pairs) or (start // batch_size) % 20 == 0:
            print(f"  scored {done}/{len(pairs)}")
    return totals, counts


def score_continuations(model, tokenizer, contexts, continuations, batch_size, device):
    pairs = [
        encode_pair(tokenizer, context, continuation)
        for context, continuation in zip(contexts, continuations)
    ]
    return sum_logprobs(model, pairs, batch_size, device)


def score_solutions(model, tokenizer, frame, batch_size, device):
    """Score both candidate solutions for each question in a single combined batch."""
    contexts = [format_context(goal) for goal in frame["goal"]]
    continuation_1 = [format_continuation(solution) for solution in frame["sol1"]]
    continuation_2 = [format_continuation(solution) for solution in frame["sol2"]]
    n = len(frame)
    totals, counts = score_continuations(
        model, tokenizer,
        contexts + contexts,
        continuation_1 + continuation_2,
        batch_size, device,
    )
    sum_1, sum_2 = totals[:n], totals[n:]
    count_1, count_2 = counts[:n], counts[n:]
    if not (np.isfinite(sum_1).all() and np.isfinite(sum_2).all()):
        raise ValueError("score_solutions produced a non-finite score")
    return sum_1, count_1, sum_2, count_2


def scores_to_frame(frame, score_1, score_2):
    pred = (np.asarray(score_2) > np.asarray(score_1)).astype(int)
    prob_sol2 = softmax_sol2(score_1, score_2)
    return prediction_frame(frame["label"], score_1, score_2, prob_sol2, pred)


def _split_metrics(frame):
    metrics = compute_metrics(frame["label"], frame["pred"])
    metrics["n"] = int(len(frame))
    return metrics


def load_lm(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The two candidate scores can be very close, sometimes differing by only
    # one word. Float16 rounding may reverse their ranking, so float32 is used.
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model, tokenizer, dtype


def choose_score(sum_accuracy, mean_accuracy):
    """Choose the scoring method with higher validation accuracy.

    If the two methods tie, choose mean. Both results are retained in metrics.
    """
    if sum_accuracy > mean_accuracy:
        return "sum"
    return "mean"


def alias_of(model_name):
    return model_name.split("/")[-1].lower()


def run_model(model_name, splits, limit, batch_size, output_dir, valid_df, test_df, device):
    started = time.perf_counter()
    model, tokenizer, dtype = load_lm(model_name, device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"{model_name} params={n_params} dtype={dtype} device={device}")

    def take(frame):
        if limit is None:
            return frame
        return frame.iloc[:limit].reset_index(drop=True)

    model_dir = Path(output_dir) / alias_of(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    valid_part = take(valid_df)
    sum_1, n_1, sum_2, n_2 = score_solutions(model, tokenizer, valid_part, batch_size, device)
    if np.any(n_1 <= 0) or np.any(n_2 <= 0):
        raise RuntimeError("continuation token count must be positive")

    valid_frames = {
        "sum": scores_to_frame(valid_part, sum_1, sum_2),
        "mean": scores_to_frame(valid_part, sum_1 / n_1, sum_2 / n_2),
    }
    valid_metrics = {name: _split_metrics(frame) for name, frame in valid_frames.items()}
    for name, frame in valid_frames.items():
        frame.to_csv(model_dir / f"valid_{name}_predictions.csv", index=False)
        print(f"  valid {name} accuracy {valid_metrics[name]['accuracy']:.4f}")

    chosen = choose_score(valid_metrics["sum"]["accuracy"], valid_metrics["mean"]["accuracy"])
    valid_frames[chosen].to_csv(model_dir / "valid_predictions.csv", index=False)
    print(f"  chosen on valid: {chosen}")

    test_metrics = None
    if "test" in splits:
        test_part = take(test_df)
        test_sum_1, test_n_1, test_sum_2, test_n_2 = score_solutions(
            model, tokenizer, test_part, batch_size, device)
        if chosen == "sum":
            test_frame = scores_to_frame(test_part, test_sum_1, test_sum_2)
        else:
            test_frame = scores_to_frame(test_part, test_sum_1 / test_n_1, test_sum_2 / test_n_2)
        test_frame.to_csv(model_dir / "test_predictions.csv", index=False)
        test_metrics = _split_metrics(test_frame)
        print(f"  test {chosen} accuracy {test_metrics['accuracy']:.4f}")

    import transformers

    payload = {
        "baseline": "b2_zero_shot_lm",
        "model_name": model_name,
        "n_params": int(n_params),
        "selection": {
            "split": "valid",
            "chosen_score": chosen,
            "tie_break": "mean",
            "valid_sum_accuracy": valid_metrics["sum"]["accuracy"],
            "valid_mean_accuracy": valid_metrics["mean"]["accuracy"],
        },
        "valid": valid_metrics,
        "test": test_metrics,
        "limit": limit,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "seconds": round(time.perf_counter() - started, 3),
        "note": "These models are approximately 100 to 500 times larger than models trained from scratch. The pretraining data may contain PIQA.",
        "versions": library_versions({"transformers": transformers.__version__}),
    }
    write_json(model_dir / "metrics.json", payload)
    print(f"wrote {model_dir}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def run(models, splits, limit, batch_size, output_dir, data_dir):
    _train_df, valid_df, test_df = piqa_to_dataframe(data_dir)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payloads = []
    for model_name in models:
        payloads.append(run_model(
            model_name, splits, limit, batch_size, output_dir, valid_df, test_df, device))
    return payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(
        models=args.models,
        splits=set(args.splits),
        limit=args.limit,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
        data_dir=DATA_DIR,
    )


if __name__ == "__main__":
    main()
