"""E5: Attention analysis for the main model, using the validation set only.

The model is the finalized E2 Transformer (2 layers, 4 heads, CLS pooling,
post-norm).
Classification uses [CLS], so only the [CLS] row is analyzed.
The co-occurrence of attention patterns and predictions indicates correlation,
not an explanation of why the model chose one option over the other
(Jain and Wallace, 2019; Wiegreffe and Pinter, 2019).

Run from the repository root:

    python -m scripts.analysis.attention_analysis --seeds 4012 4013 4014 --out reports/attention
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from scripts.data_loader.load_piqa_data import piqa_to_dataframe
from scripts.data_loader.piqa_dataloader import PIQACollator, PIQADataset
from scripts.training.run_m3 import build_model

DATA_DIR = Path("datasets/PIQA")
CONFIG_PATH = Path("experiments/E2/config.json")
CHECKPOINT_DIR = Path("experiments/E2/checkpoints")
PREDICTION_DIR = Path("experiments/E2/results/final_3seed")
BOOTSTRAP_DRAWS = 2000
RANDOM_DELETIONS = 20
MAX_EXAMPLE_TOKENS = 30
CLS_ATOL = 1e-5


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def checkpoint_path(cfg, seed):
    lr = float(cfg["learning_rates"][0])
    return CHECKPOINT_DIR / f"m3_transformer_lr{lr:g}_seed{seed}_best.pt"


def prediction_path(cfg, seed):
    lr = float(cfg["learning_rates"][0])
    return PREDICTION_DIR / f"m3_transformer_lr{lr:g}_seed{seed}_val_predictions.csv"


def load_model(cfg, seed):
    tokenizer = spm.SentencePieceProcessor(model_file=cfg["tokenizer"])
    model = build_model("transformer", tokenizer.get_piece_size(), tokenizer.pad_id(), cfg)
    path = checkpoint_path(cfg, seed)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, path


def valid_frame(tokenizer):
    """Keep only the validation set.

    The split function loads the test file as well, so discard the test data
    immediately without encoding it or performing a forward pass.
    """
    _train, valid, _test = piqa_to_dataframe(DATA_DIR)
    del _train, _test
    valid = valid.reset_index(drop=True)
    for column in ("goal", "sol1", "sol2"):
        valid[f"{column}_tokens"] = valid[column].apply(tokenizer.encode_as_ids)
    return valid


def make_loader(frame, tokenizer, batch_size):
    return DataLoader(
        PIQADataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=PIQACollator(tokenizer),
    )


@torch.no_grad()
def layer_states(model, ids, mask):
    """Extract attention weights from each layer and compute the next layer's
    input using post-norm operations.

    Returns the final-layer output of shape [B, L, D] and the attention weights
    from each layer, each with shape [B, H, L, L].
    The model must already be in eval() mode so that dropout acts as the identity.
    """
    hidden = model.embedding(ids)
    if model.enable_pos_emb:
        hidden = model.position_embedding(hidden)
    hidden = model.emb_dropout(hidden)
    padding = ~mask
    attentions = []
    for layer in model.encoder.layers:
        attn_out, attn_weights = layer.self_attn(
            hidden, hidden, hidden,
            key_padding_mask=padding,
            need_weights=True,
            average_attn_weights=False,
        )
        attentions.append(attn_weights.detach())
        hidden = layer.norm1(hidden + layer.dropout1(attn_out))
        feed_forward = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))
        hidden = layer.norm2(hidden + layer.dropout2(feed_forward))
    if model.encoder.norm is not None:
        hidden = model.encoder.norm(hidden)
    return hidden, attentions


def word_distance(left, right):
    """Calculate word-level edit distance and return the indices of unaligned words.

    Replacing one word with another counts as one edit. Under this definition,
    approximately 35.4% of validation examples have a distance of at most 1,
    and approximately 59.7% have a distance of at most 2.
    """
    if len(left) < len(right):
        # Edit distance is symmetric; backtracking proceeds in the original order,
        # so the sequences are not swapped here.
        pass
    rows = len(left) + 1
    cols = len(right) + 1
    score = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        score[i][0] = i
    for j in range(1, cols):
        score[0][j] = j
    for i, token_a in enumerate(left, 1):
        for j, token_b in enumerate(right, 1):
            substitute = score[i - 1][j - 1] + (token_a != token_b)
            score[i][j] = min(score[i - 1][j] + 1, score[i][j - 1] + 1, substitute)
    i, j = len(left), len(right)
    diff_left, diff_right = [], []
    while i or j:
        if i and j and score[i][j] == score[i - 1][j - 1] + (left[i - 1] != right[j - 1]):
            if left[i - 1] != right[j - 1]:
                diff_left.append(i - 1)
                diff_right.append(j - 1)
            i -= 1
            j -= 1
        elif i and score[i][j] == score[i - 1][j] + 1:
            diff_left.append(i - 1)
            i -= 1
        else:
            diff_right.append(j - 1)
            j -= 1
    # Backtracking proceeds from the end of each sequence to the beginning.
    # Reverse the lists to restore reading order; the sets of indices are unchanged.
    diff_left.reverse()
    diff_right.reverse()
    return score[-1][-1], diff_left, diff_right


def piece_groups(token_ids, tokenizer):
    """Group subword pieces into words using the ▁ prefix as a word boundary.

    Returns the token indices belonging to each group and the decoded words.
    """
    pieces = [tokenizer.id_to_piece(int(token_id)) for token_id in token_ids]
    groups, buffer = [], []
    for index, piece in enumerate(pieces):
        if piece.startswith("▁") and buffer:
            groups.append(buffer)
            buffer = []
        buffer.append(index)
    if buffer:
        groups.append(buffer)
    words = [tokenizer.decode([int(token_ids[index]) for index in group]).strip() for group in groups]
    return groups, words


def cls_distributions(attentions, mask):
    """Compute the [CLS] row after averaging heads at each layer and after attention rollout.

    Rollout follows Abnar and Zuidema (2020): for each layer, compute
    A' = 0.5 * A + 0.5 * I, normalize each row, and multiply the matrices
    from the first layer to the last. Padding positions are excluded from normalization.
    """
    valid = mask.to(dtype=attentions[0].dtype)
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=valid.dtype)
    eye = eye.unsqueeze(0) * valid[:, :, None] * valid[:, None, :]
    rollout = None
    last_cls = None
    for weights in attentions:
        matrix = weights.mean(dim=1) * valid[:, None, :] * valid[:, :, None]
        row_sum = matrix.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        matrix = torch.where(mask[:, :, None], matrix / row_sum, torch.zeros_like(matrix))
        mixed = 0.5 * matrix + 0.5 * eye
        mixed_sum = mixed.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        mixed = torch.where(mask[:, :, None], mixed / mixed_sum, torch.zeros_like(mixed))
        rollout = mixed if rollout is None else torch.bmm(mixed, rollout)
        last_cls = matrix[:, 0, :]
    return rollout[:, 0, :], last_cls


def region_masks(ids, mask, cls_id, sep_id, eos_id):
    """Create masks for three regions: goal, solution, and special tokens.

    Special tokens include [CLS], [SEP], and EOS (the end-of-sequence token </s>).
    """
    special = ((ids == cls_id) | (ids == sep_id) | (ids == eos_id)) & mask
    goal = torch.zeros_like(mask)
    solution = torch.zeros_like(mask)
    for row in range(ids.shape[0]):
        positions = mask[row].nonzero(as_tuple=False).flatten()
        sep_at = (ids[row, positions] == sep_id).nonzero(as_tuple=False).flatten()
        eos_at = (ids[row, positions] == eos_id).nonzero(as_tuple=False).flatten()
        sep_index = int(positions[sep_at[0]])
        eos_index = int(positions[eos_at[-1]])
        goal[row, 1:sep_index] = True
        solution[row, sep_index + 1:eos_index] = True
    goal &= mask
    solution &= mask
    return goal, solution, special


def share_of(distribution, region, mask):
    total = (distribution * mask.to(distribution.dtype)).sum(dim=-1).clamp(min=1e-12)
    return (distribution * region.to(distribution.dtype)).sum(dim=-1) / total


def token_share(region, mask):
    total = mask.sum(dim=-1).clamp(min=1).to(torch.float32)
    return region.sum(dim=-1).to(torch.float32) / total


def diff_bin(distance):
    if distance == 1:
        return "1"
    if distance == 2:
        return "2"
    if 3 <= distance <= 5:
        return "3-5"
    return ">5"


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(BOOTSTRAP_DRAWS)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def display_token(piece):
    if piece == "</s>":
        return "[EOS]"
    if piece.startswith("▁"):
        return piece[1:] or "▁"
    return piece


def mean_std(values):
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def format_pm(values):
    mean, std = mean_std(values)
    return f"{mean:.3f} ± {std:.3f}"


def format_prob(value):
    """Format very small probabilities in scientific notation and values near 1
    to six decimal places, avoiding representations rounded to 0 or 1.
    """
    if value < 1e-4:
        return f"{value:.2e}"
    if value > 0.9999:
        return f"{value:.6f}"
    return f"{value:.4f}"


@torch.no_grad()
def analyse_seed(model, tokenizer, frame, loader, seed, saved_predictions):
    cls_id = tokenizer.piece_to_id("[CLS]")
    sep_id = tokenizer.piece_to_id("[SEP]")
    eos_id = tokenizer.eos_id()
    max_cls_error = 0.0
    sum_low, sum_high = 1.0, 0.0
    pad_key_max = 0.0
    rows = []
    for batch_index, batch in enumerate(loader):
        ids_1, ids_2, mask_1, mask_2, labels = batch
        pooled = []
        distributions = []
        for ids, mask in ((ids_1, mask_1), (ids_2, mask_2)):
            hidden, attentions = layer_states(model, ids, mask)
            encoded = model.encode(ids, mask)
            max_cls_error = max(max_cls_error, (hidden[:, 0, :] - encoded).abs().max().item())
            for weights in attentions:
                valid_keys = mask[:, None, None, :]
                row_sum = (weights * valid_keys).sum(dim=-1)
                observed = row_sum[mask[:, None, :].expand_as(row_sum)]
                sum_low = min(sum_low, observed.min().item())
                sum_high = max(sum_high, observed.max().item())
                pad_key_max = max(pad_key_max, weights.masked_fill(valid_keys, 0).abs().max().item())
            rollout, last = cls_distributions(attentions, mask)
            goal, solution, special = region_masks(ids, mask, cls_id, sep_id, eos_id)
            distributions.append((ids, mask, rollout, last, goal, solution, special))
            pooled.append(encoded)
        logits = torch.cat([model.classifier(pooled[0]), model.classifier(pooled[1])], dim=1)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        batch_start = batch_index * loader.batch_size
        for offset in range(ids_1.shape[0]):
            row_id = batch_start + offset
            source = frame.iloc[row_id]
            words_1 = str(source["sol1"]).split()
            words_2 = str(source["sol2"]).split()
            distance, diff_1, diff_2 = word_distance(words_1, words_2)
            candidate_rows = []
            for side, (ids, mask, rollout, last, goal, solution, special), diff_words in (
                (0, distributions[0], diff_1),
                (1, distributions[1], diff_2),
            ):
                valid_index = mask[offset].nonzero(as_tuple=False).flatten()
                token_ids = ids[offset, valid_index].tolist()
                sep_index = token_ids.index(sep_id)
                eos_index = len(token_ids) - 1 - token_ids[::-1].index(eos_id)
                solution_ids = token_ids[sep_index + 1:eos_index]
                groups, decoded = piece_groups(solution_ids, tokenizer)
                whitespace = words_1 if side == 0 else words_2
                if decoded != whitespace:
                    raise ValueError(
                        f"Subword grouping does not match whitespace tokenization "
                        f"for row {row_id}, side {side}"
                    )
                diff_positions = []
                word_positions = []
                word_mass = []
                rollout_row = rollout[offset, valid_index]
                for word_index, group in enumerate(groups):
                    absolute = [sep_index + 1 + index for index in group]
                    word_positions.append(absolute)
                    word_mass.append(float(rollout_row[absolute].sum()))
                    if word_index in diff_words:
                        diff_positions.extend(absolute)
                diff_mask = torch.zeros(len(token_ids), dtype=torch.bool)
                if diff_positions:
                    diff_mask[diff_positions] = True
                full_rollout = torch.zeros(mask.shape[1])
                full_last = torch.zeros(mask.shape[1])
                full_rollout[valid_index] = rollout_row
                full_last[valid_index] = last[offset, valid_index]
                candidate_rows.append({
                    "length": int(mask[offset].sum()),
                    "token_ids": token_ids,
                    "rollout_share": {
                        "goal": float(share_of(rollout[offset:offset + 1], goal[offset:offset + 1], mask[offset:offset + 1])[0]),
                        "solution": float(share_of(rollout[offset:offset + 1], solution[offset:offset + 1], mask[offset:offset + 1])[0]),
                        "special": float(share_of(rollout[offset:offset + 1], special[offset:offset + 1], mask[offset:offset + 1])[0]),
                    },
                    "last_share": {
                        "goal": float(share_of(last[offset:offset + 1], goal[offset:offset + 1], mask[offset:offset + 1])[0]),
                        "solution": float(share_of(last[offset:offset + 1], solution[offset:offset + 1], mask[offset:offset + 1])[0]),
                        "special": float(share_of(last[offset:offset + 1], special[offset:offset + 1], mask[offset:offset + 1])[0]),
                    },
                    "token_share": {
                        "goal": float(token_share(goal[offset:offset + 1], mask[offset:offset + 1])[0]),
                        "solution": float(token_share(solution[offset:offset + 1], mask[offset:offset + 1])[0]),
                        "special": float(token_share(special[offset:offset + 1], mask[offset:offset + 1])[0]),
                    },
                    "diff_attention": float(rollout_row[diff_mask].sum()) if len(token_ids) else 0.0,
                    "diff_token_share": float(diff_mask.sum() / max(len(token_ids), 1)),
                    "word_positions": word_positions,
                    "word_mass": word_mass,
                    "sep_index": sep_index,
                    "diff_positions": diff_positions,
                })
            label = int(labels[offset])
            pred = int(predictions[offset])
            rows.append({
                "row_id": row_id,
                "goal": str(source["goal"]),
                "sol1": str(source["sol1"]),
                "sol2": str(source["sol2"]),
                "label": label,
                "pred": pred,
                "correct": pred == label,
                "logit_1": float(logits[offset, 0]),
                "logit_2": float(logits[offset, 1]),
                "prob_sol2": float(probabilities[offset, 1]),
                "n_diff": distance,
                "diff_words_1": [words_1[index] for index in diff_1],
                "diff_words_2": [words_2[index] for index in diff_2],
                "candidates": candidate_rows,
            })
        if batch_index % 10 == 0:
            print(f"  seed {seed} batch {batch_index + 1}/{len(loader)} cls_err {max_cls_error:.3e}")

    ours = pd.DataFrame([{
        "row_id": row["row_id"],
        "label": row["label"],
        "logit_1": row["logit_1"],
        "logit_2": row["logit_2"],
        "prob_sol2": row["prob_sol2"],
        "pred": row["pred"],
        "correct": row["correct"],
    } for row in rows])
    merged = ours.merge(saved_predictions, on="row_id", suffixes=("", "_saved"))
    pred_mismatch = int((merged["pred"] != merged["pred_saved"]).sum())
    label_mismatch = int((merged["label"] != merged["label_saved"]).sum())
    correct_mismatch = int((merged["correct"] != merged["correct_saved"]).sum())
    logit_gap = max(
        np.abs(merged["logit_1"] - merged["logit_1_saved"]).max(),
        np.abs(merged["logit_2"] - merged["logit_2_saved"]).max(),
        np.abs(merged["prob_sol2"] - merged["prob_sol2_saved"]).max(),
    )
    if pred_mismatch or label_mismatch or correct_mismatch:
        raise RuntimeError(
            f"Predictions for seed {seed} do not match val_predictions.csv: "
            f"pred {pred_mismatch}, label {label_mismatch}, correct {correct_mismatch}"
        )
    if max_cls_error >= CLS_ATOL:
        raise RuntimeError(
            f"[CLS] error for seed {seed} ({max_cls_error}) exceeds {CLS_ATOL}"
        )
    if abs(sum_low - 1) > CLS_ATOL or abs(sum_high - 1) > CLS_ATOL or pad_key_max > CLS_ATOL:
        raise RuntimeError(
            f"Attention row-sum verification failed for seed {seed}: "
            f"sum [{sum_low}, {sum_high}], pad {pad_key_max}"
        )
    print(
        f"  seed {seed} verify cls {max_cls_error:.3e} "
        f"row_sum [{sum_low:.8f}, {sum_high:.8f}] pad {pad_key_max:.3e} "
        f"logit_gap {float(logit_gap):.3e}"
    )
    return rows, {
        "seed": seed,
        "max_cls_error": max_cls_error,
        "attention_sum_min": sum_low,
        "attention_sum_max": sum_high,
        "pad_key_max": pad_key_max,
        "prediction_mismatch": 0,
        "max_abs_logit_gap": float(logit_gap),
        "n": len(rows),
    }


def share_table(all_rows):
    records = []
    for seed, rows in all_rows.items():
        for kind, key in (("rollout", "rollout_share"), ("last", "last_share")):
            for region in ("goal", "solution", "special"):
                attention, tokens = [], []
                for row in rows:
                    for candidate in row["candidates"]:
                        attention.append(candidate[key][region])
                        tokens.append(candidate["token_share"][region])
                records.append({
                    "seed": seed,
                    "source": kind,
                    "region": region,
                    "attention_mean": float(np.mean(attention)),
                    "token_mean": float(np.mean(tokens)),
                    "n_candidates": len(attention),
                })
    return pd.DataFrame(records)


def diff_table(all_rows):
    records = []
    for seed, rows in all_rows.items():
        rng = np.random.default_rng(4012 + seed)
        for outcome, wanted in (("correct", True), ("incorrect", False)):
            subset = [row for row in rows if row["n_diff"] <= 2 and row["correct"] is wanted]
            attention = [np.mean([c["diff_attention"] for c in row["candidates"]]) for row in subset]
            tokens = [np.mean([c["diff_token_share"] for c in row["candidates"]]) for row in subset]
            attention_mean, attention_low, attention_high = bootstrap_ci(attention, rng)
            token_mean, token_low, token_high = bootstrap_ci(tokens, rng)
            records.append({
                "seed": seed,
                "outcome": outcome,
                "n": len(subset),
                "attention_mean": attention_mean,
                "attention_ci_low": attention_low,
                "attention_ci_high": attention_high,
                "token_mean": token_mean,
                "token_ci_low": token_low,
                "token_ci_high": token_high,
            })
    return pd.DataFrame(records)


def error_table(all_rows):
    # Length is independent of seed. Compute quartiles using the longer of the two input sequences,
    # ensuring that the bin boundaries are identical across all three seeds.
    first_rows = next(iter(all_rows.values()))
    length_values = np.asarray([max(candidate["length"] for candidate in row["candidates"]) for row in first_rows])
    edges = np.quantile(length_values, [0, 0.25, 0.5, 0.75, 1.0])
    edges = np.maximum.accumulate(np.round(edges).astype(float))
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1
    length_codes = pd.cut(length_values, bins=edges, include_lowest=True, right=True)
    length_bin = {
        row["row_id"]: f"{int(interval.left)}–{int(interval.right)}"
        for row, interval in zip(first_rows, length_codes)
    }
    records = []
    for seed, rows in all_rows.items():
        by_id = {row["row_id"]: row for row in rows}
        for label in ["1", "2", "3-5", ">5"]:
            chosen = [row for row in rows if diff_bin(row["n_diff"]) == label]
            records.append({
                "seed": seed,
                "grouping": "n_diff",
                "bin": label,
                "n": len(chosen),
                "accuracy": float(np.mean([row["correct"] for row in chosen])) if chosen else float("nan"),
            })
        length_order = sorted(set(length_bin.values()), key=lambda label: int(str(label).split("–")[0]))
        for bin_label in length_order:
            chosen = [by_id[row_id] for row_id, name in length_bin.items() if name == bin_label]
            records.append({
                "seed": seed,
                "grouping": "length",
                "bin": bin_label,
                "n": len(chosen),
                "accuracy": float(np.mean([row["correct"] for row in chosen])) if chosen else float("nan"),
            })
    return pd.DataFrame(records)


def select_examples(rows):
    eligible = []
    for row in rows:
        if all(candidate["length"] <= MAX_EXAMPLE_TOKENS for candidate in row["candidates"]):
            item = dict(row)
            item["confidence"] = abs(row["prob_sol2"] - 0.5)
            eligible.append(item)
    eligible.sort(key=lambda item: (-item["confidence"], item["row_id"]))
    correct = [item for item in eligible if item["correct"]]
    wrong = [item for item in eligible if not item["correct"]]
    if len(correct) < 2 or len(wrong) < 2:
        raise RuntimeError("Fewer than two correct or incorrect examples have lengths of at most 30 tokens.")
    chosen = correct[:2] + wrong[:2]
    used = {item["row_id"] for item in chosen}
    minimal = [item for item in eligible if item["n_diff"] <= 2 and item["row_id"] not in used]
    minimal_correct = [item for item in minimal if item["correct"]]
    minimal_wrong = [item for item in minimal if not item["correct"]]
    if not minimal_correct or not minimal_wrong:
        raise RuntimeError("Not enough correct or incorrect examples among the minimal-difference pairs.")
    chosen.extend([minimal_correct[0], minimal_wrong[0]])
    labels = (
        ["Highest confidence and correct"] * 2
        + ["Highest confidence but incorrect"] * 2
        + ["Minimal difference and correct", "Minimal difference but incorrect"]
    )
    for item, label in zip(chosen, labels):
        item["category"] = label
    return chosen


def side_name(index):
    return "sol1" if index == 0 else "sol2"


@torch.no_grad()
def example_attention(model, tokenizer, frame, row):
    """Run the selected examples again and retain subword-level attention distributions and all four heads for plotting."""
    cls_id = tokenizer.piece_to_id("[CLS]")
    sep_id = tokenizer.piece_to_id("[SEP]")
    eos_id = tokenizer.eos_id()
    collator = PIQACollator(tokenizer)
    source = frame.iloc[row["row_id"]]
    batch = collator([{
        "goal": source["goal_tokens"],
        "sol1": source["sol1_tokens"],
        "sol2": source["sol2_tokens"],
        "label": source["label"],
    }])
    plotted = []
    for side, ids, mask in ((0, batch[0], batch[2]), (1, batch[1], batch[3])):
        _hidden, attentions = layer_states(model, ids, mask)
        rollout, last = cls_distributions(attentions, mask)
        valid_index = mask[0].nonzero(as_tuple=False).flatten()
        token_ids = ids[0, valid_index].tolist()
        pieces = [display_token(tokenizer.id_to_piece(int(token_id))) for token_id in token_ids]
        sep_index = token_ids.index(sep_id)
        heads = attentions[-1][0, :, 0, valid_index].cpu().numpy()
        plotted.append({
            "name": side_name(side),
            "tokens": pieces,
            "rollout": rollout[0, valid_index].cpu().numpy(),
            "last": last[0, valid_index].cpu().numpy(),
            "heads": heads,
            "sep_index": sep_index,
            "diff_positions": set(row["candidates"][side]["diff_positions"]),
        })
    return plotted


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def truncate_goal(goal, limit=78):
    text = " ".join(str(goal).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def draw_candidate(axis, candidate, vmax):
    image = np.vstack([candidate["rollout"], candidate["last"]])
    axis.imshow(image, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["rollout", "last layer"])
    axis.set_xticks(range(len(candidate["tokens"])))
    axis.set_xticklabels(candidate["tokens"], rotation=90, ha="center", va="top")
    axis.tick_params(axis="x", pad=1, length=2)
    axis.tick_params(axis="y", pad=1, length=2)
    for tick, index in zip(axis.get_xticklabels(), range(len(candidate["tokens"]))):
        if index in candidate["diff_positions"]:
            tick.set_color("#d73027")
    # Draw a line between [SEP] and the first subword of the solution to separate the goal from the solution.
    axis.axvline(candidate["sep_index"] + 0.5, color="white", linewidth=1.0, zorder=3)
    axis.set_ylabel(candidate["name"])


def save_figure(fig, path, tight=True):
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    fig.savefig(path.with_suffix(".pdf"), **kwargs)
    fig.savefig(path.with_suffix(".png"), dpi=300, **kwargs)
    plt.close(fig)


def plot_examples(examples, path):
    """Plot two heatmaps per example, 
    allocating sufficient vertical space to prevent the two heatmap rows 
    from collapsing under vertically rotated subword labels.
    """
    configure_matplotlib()
    n = len(examples)
    width = 6.3
    left = 1.12
    right_margin = 0.46
    cbar_w = 0.14
    cbar_gap = 0.06
    plot_w = width - left - cbar_w - cbar_gap - right_margin
    title_h = 0.42
    ax_h = 0.70
    label_h = 0.72
    between = 0.14
    between_examples = 0.20
    bottom = 0.38
    top = 0.04
    block = title_h + 2 * (ax_h + label_h) + between
    height = top + bottom + n * block + max(n - 1, 0) * between_examples
    fig = plt.figure(figsize=(width, height))

    def inches(x, y, w, h):
        return [x / width, y / height, w / width, h / height]

    cursor = height - top
    for example in examples:
        cursor -= title_h
        values = np.concatenate([
            candidate["rollout"] for candidate in example["plotted"]
        ] + [candidate["last"] for candidate in example["plotted"]])
        vmax = max(float(values.max()), 1e-3)
        heat_axes = []
        for side in (0, 1):
            cursor -= ax_h
            axis = fig.add_axes(inches(left, cursor, plot_w, ax_h))
            draw_candidate(axis, example["plotted"][side], vmax)
            heat_axes.append(axis)
            cursor -= label_h
            if side == 0:
                cursor -= between
        mark = "correct" if example["correct"] else "wrong"
        heat_axes[0].set_title(
            f"{truncate_goal(example['goal'])}\n"
            f"label {side_name(example['label'])} · pred {side_name(example['pred'])} · "
            f"P(sol2)={format_prob(example['prob_sol2'])} · {mark}",
            loc="left",
            pad=3,
        )
        cbar_bottom = heat_axes[1].get_position().y0
        cbar_height = heat_axes[0].get_position().y1 - cbar_bottom
        cax = fig.add_axes([
            (left + plot_w + cbar_gap) / width,
            cbar_bottom,
            cbar_w / width,
            cbar_height,
        ])
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=vmax), cmap="viridis")
        mappable.set_array([])
        fig.colorbar(mappable, cax=cax)
        cax.tick_params(labelsize=7)
        cursor -= between_examples
    fig.text(
        0.01, 0.012,
        "Red labels differ across the two solutions. White line sits after [SEP]. "
        "Each item has its own color scale.",
        fontsize=7, ha="left", va="bottom",
    )
    # The margins have already been allocated within the 6.3-inch figure width,
    # so do not use tight bounding boxes, 
    # which could make the exported figure exceed the two-column width.
    save_figure(fig, path, tight=False)


def content_pieces(candidate):
    """Remove special tokens so they are excluded from the plot and color scale.
    Place a vertical separator between the goal and the solution.
    """
    sep = candidate["sep_index"]
    goal, solution = [], []
    for index, (token, weight) in enumerate(zip(candidate["tokens"], candidate["rollout"])):
        if index == 0 or index == sep or token in {"[CLS]", "[SEP]", "[EOS]"}:
            continue
        piece = {
            "kind": "token",
            "text": token if token else "□",
            "value": float(weight),
            "diff": index in candidate["diff_positions"],
        }
        if index < sep:
            goal.append(piece)
        else:
            solution.append(piece)
    return goal + [{"kind": "bar", "text": "|", "value": 0.0, "diff": False}] + solution


def glyph_width(renderer, text, fontsize):
    font = FontProperties(family="DejaVu Sans", size=fontsize)
    width_px, _, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
    return width_px / renderer.dpi


def text_on_blues(value, vmax):
    rgba = plt.cm.Blues(0.0 if vmax <= 0 else value / vmax)
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return rgba, ("white" if luminance < 0.45 else "black")


def pack_highlight_line(pieces, widths, x0, x_max, gap):
    """Wrap pieces according to their widths. 
    Keep the separator with the following subword rather than leaving it at the end of a line.
    """
    lines, current, x = [], [], x0
    index = 0
    while index < len(pieces):
        piece = pieces[index]
        width = widths[index]
        take = 1
        extra = 0.0
        if piece["kind"] == "bar" and index + 1 < len(pieces):
            extra = gap + widths[index + 1]
            take = 2
        needed = width + extra
        if current and x + needed > x_max:
            lines.append(current)
            current = []
            x = x0
        for step in range(take):
            current.append((pieces[index + step], x, widths[index + step]))
            x += widths[index + step] + gap
        index += take
    if current:
        lines.append(current)
    return lines


def draw_highlight_line(axis, placed, y, box_h, fontsize, vmax):
    for piece, x, width in placed:
        if piece["kind"] == "bar":
            axis.text(
                x + width / 2, y, "|",
                ha="center", va="center", fontsize=fontsize, color="black",
                fontfamily="DejaVu Sans", clip_on=False,
            )
            continue
        rgba, ink = text_on_blues(piece["value"], vmax)
        axis.add_patch(Rectangle(
            (x, y - box_h / 2), width, box_h,
            facecolor=rgba, edgecolor="none", linewidth=0, clip_on=False,
        ))
        axis.text(
            x + width / 2, y, piece["text"],
            ha="center", va="center", fontsize=fontsize, color=ink,
            fontfamily="DejaVu Sans", clip_on=False,
        )
        if piece["diff"]:
            underline_y = y - box_h / 2 + 0.012
            axis.plot(
                [x + 0.008, x + width - 0.008],
                [underline_y, underline_y],
                color=ink, linewidth=0.8, solid_capstyle="butt", clip_on=False,
            )


def draw_scale(fig, axis, x, y, bar_w, vmax, fig_w, fig_h):
    """Draw a horizontal color scale labeled with 0 and the maximum value for the current example."""
    left_label = "0"
    right_label = f"{vmax:.2f}"
    # Leave room for the labels in the main axes; x marks the left edge of the color bar.
    bar_h = 0.07
    color_axis = fig.add_axes([
        x / fig_w,
        (y - bar_h / 2) / fig_h,
        bar_w / fig_w,
        bar_h / fig_h,
    ])
    color_axis.imshow(
        np.linspace(0, 1, 256).reshape(1, -1),
        aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
    )
    color_axis.set_xlim(-0.5, 255.5)
    color_axis.set_xticks([])
    color_axis.set_yticks([])
    for spine in color_axis.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#4d4d4d")
    axis.text(x - 0.03, y, left_label, ha="right", va="center", fontsize=7, color="black", clip_on=False)
    axis.text(x + bar_w + 0.03, y, right_label, ha="left", va="center", fontsize=7, color="black", clip_on=False)


def wrap_words(renderer, text, fontsize, max_width):
    lines, current = [], ""
    for word in text.split():
        trial = word if not current else current + " " + word
        if glyph_width(renderer, trial, fontsize) > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def plot_compact_examples(examples, path, width, max_height):
    """Plot text highlights using rollout as the background; 
    share one color scale between both candidates and display only content subwords.
    """
    configure_matplotlib()
    fontsize = 7
    pad_x = 0.012
    gap = 0.018
    box_h = 0.125
    line_pitch = 0.175
    item_gap = 0.07
    margin_x = 0.05
    margin_top = 0.05
    scale_w = 0.78
    scale_side = 0.30
    caption = (
        "Color is attention rollout on content tokens, not the last layer. "
        "[CLS], [SEP], and [EOS] are omitted and excluded from the scale. "
        "Underlined tokens differ between the two solutions."
    )
    # Measure text widths first, then determine the figure height from the actual number of lines 
    # to stay within the specified height limit.
    probe = plt.figure(figsize=(width, 2.0))
    probe.canvas.draw()
    renderer = probe.canvas.get_renderer()
    label_w = max(glyph_width(renderer, name, fontsize) for name in ("sol1", "sol2")) + 0.06
    text_x = margin_x + label_w
    text_right = width - margin_x
    scale_left = width - margin_x - scale_side - scale_w
    plans = []
    for example in examples:
        values = []
        sides = []
        for candidate in example["plotted"]:
            pieces = content_pieces(candidate)
            widths = []
            for piece in pieces:
                extra = 0.0 if piece["kind"] == "bar" else 2 * pad_x
                widths.append(glyph_width(renderer, piece["text"], fontsize) + extra)
            lines = pack_highlight_line(pieces, widths, text_x, text_right, gap)
            sides.append({"name": candidate["name"], "lines": lines})
            values.extend(piece["value"] for piece in pieces if piece["kind"] == "token")
        mark = "correct" if example["correct"] else "wrong"
        header = (
            f"label {side_name(example['label'])} · pred {side_name(example['pred'])} · "
            f"P(sol2)={format_prob(example['prob_sol2'])} · {mark}"
        )
        header_w = glyph_width(renderer, header, fontsize)
        scale_inline = scale_left - 0.22 > margin_x + header_w + 0.08
        plans.append({
            "example": example,
            "sides": sides,
            "vmax": max(max(values, default=0.0), 1e-3),
            "header": header,
            "scale_inline": scale_inline,
        })
    caption_lines = wrap_words(renderer, caption, fontsize, width - 2 * margin_x)
    plt.close(probe)

    caption_h = 0.13 * len(caption_lines) + 0.08
    body = 0.0
    for plan in plans:
        body += 0.16
        if not plan["scale_inline"]:
            body += 0.18
        body += 0.08
        for side in plan["sides"]:
            body += line_pitch * max(len(side["lines"]), 1)
        body += item_gap
    height = margin_top + body + caption_h
    if max_height is not None and height > max_height:
        raise RuntimeError(f"Compact figure height {height:.2f} inches exceeds the limit of {max_height:.2f} inches.")

    fig = plt.figure(figsize=(width, height))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.axis("off")
    cursor = height - margin_top
    for plan in plans:
        cursor -= 0.12
        axis.text(
            margin_x, cursor, plan["header"],
            ha="left", va="center", fontsize=fontsize, color="black",
            fontfamily="DejaVu Sans", clip_on=False,
        )
        if plan["scale_inline"]:
            draw_scale(fig, axis, scale_left, cursor, scale_w, plan["vmax"], width, height)
            cursor -= 0.12
        else:
            cursor -= 0.18
            draw_scale(fig, axis, scale_left, cursor, scale_w, plan["vmax"], width, height)
            cursor -= 0.12
        for side in plan["sides"]:
            for line_index, placed in enumerate(side["lines"]):
                cursor -= line_pitch
                if line_index == 0:
                    axis.text(
                        margin_x, cursor, side["name"],
                        ha="left", va="center", fontsize=fontsize, color="black",
                        fontfamily="DejaVu Sans", clip_on=False,
                    )
                draw_highlight_line(axis, placed, cursor, box_h, fontsize, plan["vmax"])
        cursor -= item_gap
    caption_y = 0.03
    for line in reversed(caption_lines):
        axis.text(
            margin_x, caption_y, line,
            ha="left", va="bottom", fontsize=fontsize, color="black",
            fontfamily="DejaVu Sans", clip_on=False,
        )
        caption_y += 0.13
    save_figure(fig, path, tight=False)


def plot_summary_compact(shares, errors, path):
    """Create a single-column Figure B with attention shares above accuracy, 
    using a total height of 3.2 inches.
    """
    configure_matplotlib()
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.0, 3.2))
    regions = ["goal", "solution", "special"]
    rollout, last, tokens = [], [], []
    rollout_err, last_err, token_err = [], [], []
    for region in regions:
        roll_values = shares[(shares["source"] == "rollout") & (shares["region"] == region)]["attention_mean"]
        last_values = shares[(shares["source"] == "last") & (shares["region"] == region)]["attention_mean"]
        token_values = shares[(shares["source"] == "rollout") & (shares["region"] == region)]["token_mean"]
        rollout.append(roll_values.mean())
        last.append(last_values.mean())
        tokens.append(token_values.mean())
        rollout_err.append(roll_values.std(ddof=1))
        last_err.append(last_values.std(ddof=1))
        token_err.append(token_values.std(ddof=1))
    x = np.arange(len(regions))
    bar_w = 0.24
    axes[0].bar(x - bar_w, tokens, bar_w, yerr=token_err, label="token share", color="#bdbdbd", capsize=1.5, error_kw={"elinewidth": 0.6})
    axes[0].bar(x, rollout, bar_w, yerr=rollout_err, label="rollout", color="#2c7fb8", capsize=1.5, error_kw={"elinewidth": 0.6})
    axes[0].bar(x + bar_w, last, bar_w, yerr=last_err, label="last layer", color="#7fcdbb", capsize=1.5, error_kw={"elinewidth": 0.6})
    axes[0].set_xticks(x, regions)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_ylabel("share")
    axes[0].set_title("Where [CLS] attends", pad=2)
    # The tallest bar is approximately 0.75. 
    # Place the legend inside the top of the axes to avoid overlapping with the title.
    axes[0].legend(
        frameon=False, ncol=3, fontsize=7, loc="upper center",
        borderaxespad=0.05,
        handlelength=0.9, columnspacing=0.7, handletextpad=0.3,
    )
    axes[0].tick_params(length=2, pad=1)

    diff_rows = errors[errors["grouping"] == "n_diff"]
    bins = ["1", "2", "3-5", ">5"]
    means, stds = [], []
    for label in bins:
        values = diff_rows[diff_rows["bin"] == label]["accuracy"]
        means.append(values.mean())
        stds.append(values.std(ddof=1))
    axes[1].bar(bins, means, yerr=stds, color="#2c7fb8", capsize=1.5, error_kw={"elinewidth": 0.6})
    axes[1].axhline(0.5, color="#737373", linewidth=0.6, linestyle="--")
    axes[1].set_ylim(0.45, 0.72)
    axes[1].set_yticks([0.5, 0.6, 0.7])
    axes[1].set_xlabel("word-level edit distance")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy by solution difference", pad=2)
    axes[1].tick_params(length=2, pad=1)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.14, hspace=0.72)
    save_figure(fig, path, tight=False)


def plot_heads(example, path):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), squeeze=False)
    for side in (0, 1):
        candidate = example["plotted"][side]
        image = candidate["heads"]
        axis = axes[0, side]
        axis.imshow(image, aspect="auto", cmap="viridis", vmin=0, vmax=max(float(image.max()), 1e-3), interpolation="nearest")
        axis.set_yticks(range(image.shape[0]))
        axis.set_yticklabels([f"head {index + 1}" for index in range(image.shape[0])])
        axis.set_xticks(range(len(candidate["tokens"])))
        axis.set_xticklabels(candidate["tokens"], rotation=90)
        for tick, index in zip(axis.get_xticklabels(), range(len(candidate["tokens"]))):
            if index in candidate["diff_positions"]:
                tick.set_color("#d73027")
        axis.axvline(candidate["sep_index"] + 0.5, color="white", linewidth=1.0, zorder=3)
        axis.set_title(candidate["name"])
    mark = "correct" if example["correct"] else "wrong"
    fig.suptitle(
        f"Last layer, four heads · {truncate_goal(example['goal'], 70)} · {mark}",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_summary(shares, errors, path):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    regions = ["goal", "solution", "special"]
    rollout = []
    last = []
    tokens = []
    rollout_err, last_err, token_err = [], [], []
    for region in regions:
        roll_values = shares[(shares["source"] == "rollout") & (shares["region"] == region)]["attention_mean"]
        last_values = shares[(shares["source"] == "last") & (shares["region"] == region)]["attention_mean"]
        token_values = shares[(shares["source"] == "rollout") & (shares["region"] == region)]["token_mean"]
        rollout.append(roll_values.mean())
        last.append(last_values.mean())
        tokens.append(token_values.mean())
        rollout_err.append(roll_values.std(ddof=1))
        last_err.append(last_values.std(ddof=1))
        token_err.append(token_values.std(ddof=1))
    x = np.arange(len(regions))
    width = 0.24
    axes[0].bar(x - width, tokens, width, yerr=token_err, label="token share", color="#bdbdbd", capsize=2)
    axes[0].bar(x, rollout, width, yerr=rollout_err, label="rollout", color="#2c7fb8", capsize=2)
    axes[0].bar(x + width, last, width, yerr=last_err, label="last layer", color="#7fcdbb", capsize=2)
    axes[0].set_xticks(x, regions)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("share")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Where [CLS] attends")

    diff_rows = errors[errors["grouping"] == "n_diff"]
    bins = ["1", "2", "3-5", ">5"]
    means, stds = [], []
    for label in bins:
        values = diff_rows[diff_rows["bin"] == label]["accuracy"]
        means.append(values.mean())
        stds.append(values.std(ddof=1))
    axes[1].bar(bins, means, yerr=stds, color="#2c7fb8", capsize=2)
    axes[1].axhline(0.5, color="#737373", linewidth=0.6, linestyle="--")
    axes[1].set_ylim(0.45, max(0.7, max(means) + 0.05))
    axes[1].set_xlabel("word-level edit distance")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title("Accuracy by solution difference")
    fig.tight_layout()
    save_figure(fig, path)


@torch.no_grad()
def intervention(model, tokenizer, rows, seed):
    """删掉正确答案那一侧方案里 rollout 最高的词，对照随机删一个词。"""
    rng = np.random.default_rng(4012 + seed)
    collator_pad = tokenizer.pad_id()
    jobs = []
    base_probs = []
    top_is_diff = []
    for row in rows:
        side = row["label"]
        candidate = row["candidates"][side]
        if not candidate["word_positions"]:
            continue
        masses = np.asarray(candidate["word_mass"], dtype=float)
        top = int(masses.argmax())
        random_picks = rng.integers(0, len(masses), size=RANDOM_DELETIONS)
        other = row["candidates"][1 - side]["token_ids"]
        base_probs.append(row["prob_sol2"] if side == 1 else 1.0 - row["prob_sol2"])
        top_is_diff.append(candidate["word_positions"][top][0] in candidate["diff_positions"] or any(
            position in candidate["diff_positions"] for position in candidate["word_positions"][top]
        ))
        jobs.append((row["label"], candidate["token_ids"], other, side, candidate["word_positions"], top, random_picks))

    def run_deleted(specs):
        # specs: list of (label, ids_left, ids_right)
        left = [torch.tensor(item[1], dtype=torch.long) for item in specs]
        right = [torch.tensor(item[2], dtype=torch.long) for item in specs]
        ids_1 = torch.nn.utils.rnn.pad_sequence(left, batch_first=True, padding_value=collator_pad)
        ids_2 = torch.nn.utils.rnn.pad_sequence(right, batch_first=True, padding_value=collator_pad)
        mask_1 = ids_1 != collator_pad
        mask_2 = ids_2 != collator_pad
        pooled_1 = model.encode(ids_1, mask_1)
        pooled_2 = model.encode(ids_2, mask_2)
        logits = torch.cat([model.classifier(pooled_1), model.classifier(pooled_2)], dim=1)
        probabilities = torch.softmax(logits, dim=1)
        labels = torch.tensor([item[0] for item in specs])
        return probabilities[torch.arange(len(specs)), labels].cpu().numpy()

    top_delta = []
    random_delta = []
    batch = []
    batch_base = []
    batch_kind = []

    def flush():
        if not batch:
            return
        probs = run_deleted(batch)
        for prob, base, kind in zip(probs, batch_base, batch_kind):
            delta = float(prob - base)
            if kind == "top":
                top_delta.append(delta)
            else:
                random_delta.append(delta)
        batch.clear()
        batch_base.clear()
        batch_kind.clear()

    for (label, token_ids, other, side, positions, top, picks), base in zip(jobs, base_probs):
        def deleted(word_index):
            drop = set(positions[word_index])
            kept = [token for index, token in enumerate(token_ids) if index not in drop]
            if side == 0:
                return label, kept, other
            return label, other, kept

        batch.append(deleted(top))
        batch_base.append(base)
        batch_kind.append("top")
        for pick in picks:
            batch.append(deleted(int(pick)))
            batch_base.append(base)
            batch_kind.append("random")
            if len(batch) >= 32:
                flush()
    flush()
    random_by_example = np.asarray(random_delta).reshape(len(jobs), RANDOM_DELETIONS).mean(axis=1)
    return {
        "seed": seed,
        "n": len(jobs),
        "top_mean": float(np.mean(top_delta)),
        "random_mean": float(np.mean(random_by_example)),
        "top_is_diff_rate": float(np.mean(top_is_diff)),
    }


def example_notes(example, goal_baseline):
    """按这道题自己的占比写一两句，不把注意力说成原因。"""
    goal_n = example["plotted"][0]["sep_index"] - 1
    goal_share = float(np.mean([candidate["rollout_share"]["goal"] for candidate in example["candidates"]]))
    goal_token = float(np.mean([candidate["token_share"]["goal"] for candidate in example["candidates"]]))
    diff_share = float(np.mean([candidate["diff_attention"] for candidate in example["candidates"]]))
    diff_tokens = float(np.mean([candidate["diff_token_share"] for candidate in example["candidates"]]))
    notes = []
    if goal_n <= 3:
        notes.append(f"goal 在序列里只有 {goal_n} 个子词。")
    if example["n_diff"] <= 2:
        if diff_share > diff_tokens + 0.05:
            relation = "高于"
        elif diff_share + 0.05 < diff_tokens:
            relation = "低于"
        else:
            relation = "接近"
        notes.append(
            f"两个方案不同的词，rollout {diff_share:.3f}，token 占比 {diff_tokens:.3f}，{relation}长度占比。"
        )
    else:
        notes.append(
            f"词级距离是 {example['n_diff']}，两个方案大部分词都不同。"
            f"差异词 rollout {diff_share:.3f}，token 占比 {diff_tokens:.3f}，这个比例铺在整个方案上。"
        )
    if goal_share < goal_token - 0.05:
        clause = f"这道题的 goal rollout {goal_share:.3f}，低于它自己的 token 占比 {goal_token:.3f}"
    elif goal_share > goal_token + 0.05:
        clause = f"这道题的 goal rollout {goal_share:.3f}，高于它自己的 token 占比 {goal_token:.3f}"
    else:
        clause = f"这道题的 goal rollout {goal_share:.3f}，和它的 token 占比 {goal_token:.3f} 接近"
    if goal_share > goal_baseline + 0.10:
        clause += f"，也高于验证集 goal rollout 的平均 {goal_baseline:.3f}"
    notes.append(clause + "。")
    return notes[:2]


def heads_note(example):
    parts = []
    for candidate in example["plotted"]:
        sep = candidate["sep_index"]
        peaks = []
        goal_mass = []
        for index, row in enumerate(candidate["heads"]):
            position = int(np.argmax(row))
            peaks.append(f"头{index + 1}「{candidate['tokens'][position]}」")
            goal_mass.append(float(row[:sep].sum()))
        parts.append(
            candidate["name"] + "：" + "，".join(peaks)
            + f"（SEP 左侧权重之和平均 {float(np.mean(goal_mass)):.2f}，摊在多个子词上）"
        )
    return "最后一层四个头的最大值不在同一个子词上。" + " ".join(parts) + "。"


def write_summary(path, verifies, shares, diffs, errors, examples, interventions):
    lines = [
        "# E5 注意力分析",
        "",
        "主模型是 E2 定稿的 Transformer：2 层、4 头、embedding 100、CLS pooling、post-norm、dropout 0.3。",
        "权重是 `experiments/E2/checkpoints/` 里三个 seed 的 `best.pt`。全部数字来自验证集 1606 题，没有使用测试集。",
        "分类只用位置 0 的 [CLS]，下面的注意力都是 [CLS] 这一行。rollout 按 Abnar and Zuidema (2020)，先对头取平均。",
        "",
        "注意力分布和做对做错同时出现，只能说明相关，不能说明模型是因为看到这些 token 才做出选择（Jain and Wallace, 2019; Wiegreffe and Pinter, 2019）。",
        "",
        "## 验证",
        "",
    ]
    for item in verifies:
        lines.append(
            f"- seed {item['seed']}: [CLS] 和 `encode()` 的最大绝对误差 {item['max_cls_error']:.3e}；"
            f"有效 query 行在有效 token 上的和落在 [{item['attention_sum_min']:.8f}, {item['attention_sum_max']:.8f}]，"
            f"padding 键上的最大值 {item['pad_key_max']:.3e}；"
            f"pred / label / correct 和 `val_predictions.csv` 一致，logit 与概率的最大绝对差 {item['max_abs_logit_gap']:.3e}。"
        )
    lines.extend(["", "## goal、方案、特殊符号", ""])
    lines.append("每个数是三个 seed 的均值 ± 样本标准差。一道题的两个候选各算一次，再对全部验证题取平均。token 占比是这三组的长度占比。")
    lines.append("")
    lines.append("| 区域 | token 占比 | rollout | 最后一层 |")
    lines.append("| --- | --- | --- | --- |")
    for region in ("goal", "solution", "special"):
        token_values = shares[(shares["source"] == "rollout") & (shares["region"] == region)]["token_mean"]
        roll_values = shares[(shares["source"] == "rollout") & (shares["region"] == region)]["attention_mean"]
        last_values = shares[(shares["source"] == "last") & (shares["region"] == region)]["attention_mean"]
        lines.append(f"| {region} | {format_pm(token_values)} | {format_pm(roll_values)} | {format_pm(last_values)} |")
    lines.extend([
        "",
        "E4 的 A2-a 在评估时去掉 goal，验证准确率下降约 2.4 个点，三个 seed 里改变的预测大约是 15.5%–18.8%（`experiments/E4/README4.md`）。",
        (
            f"goal 的 rollout 是 {format_pm(shares[(shares['source'] == 'rollout') & (shares['region'] == 'goal')]['attention_mean'])}，"
            f"最后一层是 {format_pm(shares[(shares['source'] == 'last') & (shares['region'] == 'goal')]['attention_mean'])}，"
            f"都低于 token 占比 {format_pm(shares[(shares['source'] == 'rollout') & (shares['region'] == 'goal')]['token_mean'])}。"
            f"方案的 rollout 是 {format_pm(shares[(shares['source'] == 'rollout') & (shares['region'] == 'solution')]['attention_mean'])}，"
            f"接近 token 占比 {format_pm(shares[(shares['source'] == 'rollout') & (shares['region'] == 'solution')]['token_mean'])}；"
            f"最后一层是 {format_pm(shares[(shares['source'] == 'last') & (shares['region'] == 'solution')]['attention_mean'])}，高于长度占比。"
            f"特殊符号的 rollout 是 {format_pm(shares[(shares['source'] == 'rollout') & (shares['region'] == 'special')]['attention_mean'])}，"
            f"高于 token 占比 {format_pm(shares[(shares['source'] == 'rollout') & (shares['region'] == 'special')]['token_mean'])}；"
            f"最后一层只有 {format_pm(shares[(shares['source'] == 'last') & (shares['region'] == 'special')]['attention_mean'])}。"
        ),
        "rollout 在每一层用 A' = 0.5·A + 0.5·I，单位阵会把一部分质量留在 [CLS] 上，所以特殊符号这一格偏高。最后一层没有加单位阵，特殊符号低于它的长度占比，方案则高于长度占比。",
        "goal 在两种算法里都低于自己的长度占比。这和 A2-a 里大约 15%–19% 的预测发生改变是同一方向的相关。注意力权重不是因果解释。",
        "",
        "## 差异词",
        "",
        "两个方案按空格分词后做词级编辑距离。换成一个词记距离 1。距离不超过 2 的题里，把不同的词映射到以 ▁ 开头的 SentencePiece 词，再看 rollout 落在这些子词上的比例。",
        "每题先对两个候选取平均，再在做对和做错里分开。括号里是该 seed 的 95% bootstrap 置信区间（2000 次百分位；Generator 种子是 4012 + seed）。跨 seed 再报均值 ± 标准差。",
        "",
    ])
    for outcome, title in (("correct", "做对"), ("incorrect", "做错")):
        part = diffs[diffs["outcome"] == outcome]
        lines.append(
            f"- {title}（每 seed 约 {int(part['n'].mean())} 题）：rollout {format_pm(part['attention_mean'])}，"
            f"token 占比 {format_pm(part['token_mean'])}。"
        )
        for _, item in part.iterrows():
            lines.append(
                f"  - seed {int(item['seed'])}: rollout {item['attention_mean']:.3f} "
                f"[{item['attention_ci_low']:.3f}, {item['attention_ci_high']:.3f}]，"
                f"token {item['token_mean']:.3f} [{item['token_ci_low']:.3f}, {item['token_ci_high']:.3f}]，n={int(item['n'])}"
            )
    correct_part = diffs[diffs["outcome"] == "correct"]
    incorrect_part = diffs[diffs["outcome"] == "incorrect"]
    overlapped = []
    for seed in correct_part["seed"].tolist():
        correct_row = correct_part[correct_part["seed"] == seed].iloc[0]
        incorrect_row = incorrect_part[incorrect_part["seed"] == seed].iloc[0]
        overlapped.append(
            correct_row["attention_ci_low"] <= incorrect_row["attention_ci_high"]
            and incorrect_row["attention_ci_low"] <= correct_row["attention_ci_high"]
        )
    if overlapped and all(overlapped):
        lines.append("")
        lines.append(
            "三个 seed 上，做对和做错的 rollout 置信区间都有重叠。"
            f"做对的均值是 {format_pm(correct_part['attention_mean'])}，做错是 {format_pm(incorrect_part['attention_mean'])}，"
            "两边都高于各自的 token 占比。"
        )
    lines.extend(["", "## 错误分析", ""])
    lines.append("| 分组 | 组 | 题数 | 准确率 |")
    lines.append("| --- | --- | --- | --- |")
    for grouping in ("n_diff", "length"):
        part = errors[errors["grouping"] == grouping]
        for bin_label, group in part.groupby("bin", sort=False):
            lines.append(
                f"| {grouping} | {bin_label} | {int(group['n'].iloc[0])} | {format_pm(group['accuracy'])} |"
            )
    diff_part = errors[errors["grouping"] == "n_diff"]
    diff_means = {
        label: float(diff_part[diff_part["bin"] == label]["accuracy"].mean())
        for label in ["1", "2", "3-5", ">5"]
    }
    acc_bits = [f"距离 {label} 为 {format_pm(diff_part[diff_part['bin'] == label]['accuracy'])}" for label in ["1", "2", "3-5", ">5"]]
    low_bin = min(diff_means, key=diff_means.get)
    high_bin = max(diff_means, key=diff_means.get)
    lines.append("")
    lines.append(
        "准确率没有随着词级距离变大而下降：" + "，".join(acc_bits) + f"。其中 {low_bin} 最低，{high_bin} 最高，误差条有重叠。"
    )
    length_part = errors[errors["grouping"] == "length"]
    length_bits = []
    length_means = []
    for bin_label, group in length_part.groupby("bin", sort=False):
        length_bits.append(f"{bin_label}（n={int(group['n'].iloc[0])}，{format_pm(group['accuracy'])}）")
        length_means.append(float(group["accuracy"].mean()))
    length_sentence = (
        "序列长度按两个候选里较长的一段取四分位，从短到长是 "
        + "，".join(length_bits)
        + "。最长一组的右端拉得很长，因为验证集长度右偏。"
    )
    if len(length_means) >= 3 and length_means[-1] < min(length_means[1:-1]):
        length_sentence += "这一组低于中间的组。"
    lines.append(length_sentence)
    if interventions:
        lines.extend(["", "## 删词（可选检查）", ""])
        lines.append("只改正确答案那一侧的方案：删掉 rollout 最高的词，或者随机删一个词（20 次取平均）。数字是正确选项概率的变化，三个 seed 各自的平均值再汇总。")
        lines.append(f"- 删最高词：{format_pm([item['top_mean'] for item in interventions])}")
        lines.append(f"- 随机删一个词：{format_pm([item['random_mean'] for item in interventions])}")
        lines.append(f"- 最高词本身就是差异词的比例：{format_pm([item['top_is_diff_rate'] for item in interventions])}")
        top_mean = float(np.mean([item["top_mean"] for item in interventions]))
        random_mean = float(np.mean([item["random_mean"] for item in interventions]))
        diff_rate = float(np.mean([item["top_is_diff_rate"] for item in interventions]))
        if top_mean < random_mean - 0.01:
            lines.append("删掉 rollout 最高的词时，正确选项概率下降得比随机删词更多。")
        if diff_rate < 0.5:
            lines.append("只改了正确答案那一侧的方案，另一侧保持原样。最高词里不到一半本身就是差异词，所以多数时候删掉的不是两个方案的对比词。")
        else:
            lines.append("只改了正确答案那一侧的方案，另一侧保持原样。最高词里至少一半本身就是差异词。")
        lines.append("概率变了，说明这些词和这次输出有关。这仍然不是注意力导致了选择的证明（Jain and Wallace, 2019; Wiegreffe and Pinter, 2019）。")
    lines.extend(["", "## 例子", ""])
    lines.append("图只使用 seed 4012。先限制两个候选都不超过 30 个 token，再按 |P(sol2) − 0.5| 从高到低取，并列时取 row_id 小的。")
    lines.append("")
    goal_baseline = float(shares[(shares["source"] == "rollout") & (shares["region"] == "goal")]["attention_mean"].mean())
    for example_index, example in enumerate(examples):
        diff_text = " / ".join(example["diff_words_1"]) or "（sol1 没有独有的词）"
        other = " / ".join(example["diff_words_2"]) or "（sol2 没有独有的词）"
        goal_share = np.mean([candidate["rollout_share"]["goal"] for candidate in example["candidates"]])
        solution_share = np.mean([candidate["rollout_share"]["solution"] for candidate in example["candidates"]])
        special_share = np.mean([candidate["rollout_share"]["special"] for candidate in example["candidates"]])
        diff_share = np.mean([candidate["diff_attention"] for candidate in example["candidates"]])
        diff_tokens = np.mean([candidate["diff_token_share"] for candidate in example["candidates"]])
        lines.append(
            f"- row {example['row_id']}（{example['类别']}）：{truncate_goal(example['goal'], 90)}"
        )
        lines.append(
            f"  正确答案 {side_name(example['label'])}，预测 {side_name(example['pred'])}，"
            f"P(sol2)={format_prob(example['prob_sol2'])}。差异词是「{diff_text}」对「{other}」。"
        )
        lines.append(
            f"  两个候选平均后，rollout 在 goal {goal_share:.3f}、方案 {solution_share:.3f}、特殊符号 {special_share:.3f}；"
            f"差异词的 rollout {diff_share:.3f}，token 占比 {diff_tokens:.3f}。"
        )
        for note in example_notes(example, goal_baseline):
            lines.append(f"  {note}")
        if example_index == 0 and example.get("plotted"):
            lines.append(f"  {heads_note(example)}")
    lines.extend([
        "",
        "## 图",
        "",
        "- `figA_examples`：把握最大的做对、做错各 1 题，加上最小差异对的做对和做错。热图，较高。",
        "- `figA_examples_all6`：上面的规则选出的全部 6 题。",
        "- `figA_examples_compact`：报告用，双栏 6.3 英寸。同样 4 题，文字高亮，只画 rollout。",
        "- `figA_examples_single_compact`：单栏 3.0 英寸。只放 row 1142 和 1296。",
        "- `figB_share_and_accuracy`：区域占比，以及准确率随词级距离的变化。双栏。",
        "- `figB_share_and_accuracy_compact`：报告用，单栏 3.0 英寸，两个小图上下排。",
        "- `fig_heads_last_layer`：附图。把握最大且做对的那一题，最后一层 4 个头。",
        "",
    ])
    path.write_text("\n".join(lines))


def load_saved_predictions(cfg, seed):
    saved = pd.read_csv(prediction_path(cfg, seed))
    saved["correct"] = saved["correct"].astype(str).str.lower().isin(["true", "1"])
    saved["pred"] = saved["pred"].astype(int)
    saved["label"] = saved["label"].astype(int)
    saved = saved.rename(columns={
        "label": "label_saved",
        "logit_1": "logit_1_saved",
        "logit_2": "logit_2_saved",
        "prob_sol2": "prob_sol2_saved",
        "pred": "pred_saved",
        "correct": "correct_saved",
    })
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[4012, 4013, 4014])
    parser.add_argument("--out", default="reports/attention")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    all_rows = {}
    verifies = []
    frame = None
    model_4012 = None
    tokenizer_4012 = None
    for seed in args.seeds:
        print(f"seed {seed}")
        model, tokenizer, path = load_model(cfg, seed)
        print(f"  checkpoint {path}")
        if frame is None:
            frame = valid_frame(tokenizer)
        loader = make_loader(frame, tokenizer, args.batch_size)
        rows, verify = analyse_seed(model, tokenizer, frame, loader, seed, load_saved_predictions(cfg, seed))
        all_rows[seed] = rows
        verifies.append(verify)
        if seed == 4012:
            model_4012 = model
            tokenizer_4012 = tokenizer
        else:
            del model
    if 4012 not in all_rows:
        raise RuntimeError("例子图需要 seed 4012")

    shares = share_table(all_rows)
    diffs = diff_table(all_rows)
    errors = error_table(all_rows)
    shares.to_csv(out / "goal_solution_share.csv", index=False)
    diffs.to_csv(out / "diff_token_attention.csv", index=False)
    errors.to_csv(out / "error_by_diff.csv", index=False)

    selected = select_examples(all_rows[4012])
    for example in selected:
        example["plotted"] = example_attention(model_4012, tokenizer_4012, frame, example)
    selected_frame = pd.DataFrame([{
        "row_id": example["row_id"],
        "类别": example["类别"],
        "goal": example["goal"],
        "sol1": example["sol1"],
        "sol2": example["sol2"],
        "正确答案": side_name(example["label"]),
        "预测": side_name(example["pred"]),
        "prob_sol2": example["prob_sol2"],
    } for example in selected])
    selected_frame.to_csv(out / "selected_examples.csv", index=False)

    report_examples = [selected[0], selected[2], selected[4], selected[5]]
    plot_examples(report_examples, out / "figA_examples")
    plot_examples(selected, out / "figA_examples_all6")
    plot_summary(shares, errors, out / "figB_share_and_accuracy")
    plot_heads(selected[0], out / "fig_heads_last_layer")
    by_id = {example["row_id"]: example for example in selected}
    plot_compact_examples(
        report_examples, out / "figA_examples_compact", width=6.3, max_height=3.5,
    )
    plot_compact_examples(
        [by_id[1142], by_id[1296]],
        out / "figA_examples_single_compact",
        width=3.0,
        max_height=None,
    )
    plot_summary_compact(shares, errors, out / "figB_share_and_accuracy_compact")

    interventions = []
    for seed in args.seeds:
        print(f"intervention seed {seed}")
        if seed == 4012:
            model = model_4012
            tokenizer = tokenizer_4012
        else:
            model, tokenizer, _path = load_model(cfg, seed)
        interventions.append(intervention(model, tokenizer, all_rows[seed], seed))
        print(
            f"  top {interventions[-1]['top_mean']:.4f} "
            f"random {interventions[-1]['random_mean']:.4f}"
        )
    pd.DataFrame(interventions).to_csv(out / "intervention.csv", index=False)
    write_summary(out / "summary.md", verifies, shares, diffs, errors, selected, interventions)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
