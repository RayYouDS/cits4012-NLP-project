"""B2：零样本语言模型打分。

这些模型比我们从零训练的模型大约 100 到 500 倍。比较的是训练方式
（从零训练 / 不训练、只靠预训练），不是同等大小的模型。
预训练语料可能已经包含 PIQA，存在数据污染，报告里要写成局限。

使用各模型自带的分词器，不能把 piqa_bpe_v3 的 id 送进去。

从仓库根目录运行：

    python -m scripts.baselines.b2_zero_shot_lm

只在验证集上比较 sum 和 mean，选定一种之后，测试集只评这一种。
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
    """分开编码再拼接。只对 continuation 的 token 计分。"""
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if len(context_ids) == 0:
        raise ValueError("context produced no tokens")
    if len(continuation_ids) == 0:
        raise ValueError("continuation produced no tokens")
    return context_ids, continuation_ids


def _continuation_sum(log_probs, input_ids, context_len, continuation_len):
    """logits[t] 预测位置 t+1。continuation 的第一个 token 由 context 的最后一个位置预测。"""
    token_ids = input_ids[context_len:context_len + continuation_len]
    chosen = log_probs[context_len - 1:context_len + continuation_len - 1]
    token_logprob = chosen.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return float(token_logprob.sum().item()), continuation_len


def _forward_log_probs(model, input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return torch.log_softmax(output.logits.float(), dim=-1)


@torch.no_grad()
def sum_logprobs(model, pairs, batch_size, device):
    """pairs: (context_ids, continuation_ids)。返回 continuation 的对数概率之和，以及 token 数。

    GPU 上按 batch 右侧 padding，attention mask 把 pad 标成 0。真实 token 仍在位置 0..L-1。
    CPU 的 float32 矩阵乘会随 batch 形状改变，同一条的 sum 在 batch=1 和 batch=8 之间
    可以差到大约 3e-4。CPU 因此逐条前向，两种 batch 大小得到同一个分数。
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
    """同一道题的两个方案拼进一个 batch 序列，再拆开。"""
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
    return totals[:n], counts[:n], totals[n:], counts[n:]


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
    use_fp16 = device.type == "cuda" and "Qwen" in model_name
    dtype = torch.float16 if use_fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model, tokenizer, dtype


def choose_score(sum_accuracy, mean_accuracy):
    """验证集准确率更高的那种。打平时用 mean，两种结果都写进 metrics。"""
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
        "note": "模型比从零训练的模型大约 100 到 500 倍。预训练数据可能包含 PIQA。",
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
