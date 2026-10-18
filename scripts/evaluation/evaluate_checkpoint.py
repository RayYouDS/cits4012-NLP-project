"""Load a trained checkpoint and evaluate it once on the validation or test set.

Run from the repository root:

    python -m scripts.evaluation.evaluate_checkpoint \
        --config config/m3_params.json \
        --model transformer \
        --checkpoint path/to/best.pt \
        --split valid \
        --output-dir reports/eval/run_name

The model is constructed using scripts.training.run_m3.build_model and the
same configuration fields used during training.

The goal is included by default. For the A2-a ablation, pass
--no-include-goal to use the input format [CLS] [SEP] solution [EOS].

Both validation and test DataLoaders use shuffle=False. The test set should
only be evaluated after the model has been finalised.
"""


import argparse
import json
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence

from scripts.baselines.results import library_versions, write_json
from scripts.data_loader.piqa_dataloader import PIQACollator, PIQADataset, tokenize_piqa_dataframe
from scripts.evaluation.evaluate import evaluate
from scripts.training.run_m3 import build_model, make_loader

DATA_DIR = Path("datasets/PIQA")


class OptionalGoalCollator:
    """
    Optionally exclude the goal from the input sequence without modifying
    the existing PIQACollator implementation.
    """

    def __init__(self, tokenizer, include_goal=True):
        self.include_goal = include_goal
        self.base = PIQACollator(tokenizer)
        self.pad_id = tokenizer.pad_id()
        self.cls_id = tokenizer.piece_to_id("[CLS]")
        self.sep_id = tokenizer.piece_to_id("[SEP]")
        self.eos_id = tokenizer.eos_id()
        if self.cls_id == tokenizer.unk_id() or self.sep_id == tokenizer.unk_id():
            raise ValueError("[CLS] or [SEP] is not defined in tokenizer")

    def __call__(self, batch):
        if self.include_goal:
            return self.base(batch)

        input_1, input_2, labels = [], [], []
        for item in batch:
            input_1.append(self._sequence(item["sol1"]))
            input_2.append(self._sequence(item["sol2"]))
            labels.append(item["label"])
        input_1 = pad_sequence(input_1, batch_first=True, padding_value=self.pad_id)
        input_2 = pad_sequence(input_2, batch_first=True, padding_value=self.pad_id)
        mask_1 = input_1 != self.pad_id
        mask_2 = input_2 != self.pad_id
        labels = torch.tensor(labels, dtype=torch.long)
        return input_1, input_2, mask_1, mask_2, labels

    def _sequence(self, solution_tokens):
        return torch.tensor(
            [self.cls_id, self.sep_id, *solution_tokens, self.eos_id],
            dtype=torch.long,
        )


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run(cfg, model_name, checkpoint, split, output_dir, include_goal=True):
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor(model_file=cfg["tokenizer"])
    _train_df, valid_df, test_df = tokenize_piqa_dataframe(DATA_DIR, tokenizer)
    split_df = {"valid": valid_df, "test": test_df}[split].reset_index(drop=True)

    collator = OptionalGoalCollator(tokenizer, include_goal=include_goal)
    loader = make_loader(
        split_df, collator, int(cfg["batch_size"]),
        num_workers=0, pin_memory=device.type == "cuda", shuffle=False,
    )
    model = build_model(
        model_name, tokenizer.get_piece_size(), tokenizer.pad_id(), cfg,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    metrics, predictions = evaluate(model, loader, device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    n_params = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "split": split,
        "include_goal": bool(include_goal),
        "n_params": int(n_params),
        "selection": None,
        "metrics": metrics,
        "n": int(len(predictions)),
        "seconds": round(time.perf_counter() - started, 3),
        "versions": library_versions(),
    }
    write_json(output_dir / "metrics.json", payload)
    print(
        f"{split} accuracy {metrics['accuracy']:.6f}  f1 {metrics['f1']:.6f}  "
        f"n {len(predictions)}  include_goal {include_goal}"
    )
    print(f"wrote {predictions_path}")
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=("gru", "lstm", "transformer"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-goal", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(
        cfg=load_config(args.config),
        model_name=args.model,
        checkpoint=Path(args.checkpoint),
        split=args.split,
        output_dir=Path(args.output_dir),
        include_goal=args.include_goal,
    )


if __name__ == "__main__":
    main()
