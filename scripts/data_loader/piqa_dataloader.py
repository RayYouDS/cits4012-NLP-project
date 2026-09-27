from scripts.data_loader.load_piqa_data import piqa_to_dataframe
from pathlib import Path
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ================ batch tokenizer ========================
def tokenize_piqa_dataframe(base_path: Path, tokenizer: spm.SentencePieceProcessor):
    if not isinstance(tokenizer, spm.SentencePieceProcessor):
        raise TypeError('Tokenizer must be a spm.SentencePieceProcessor object')

    train_dataset, valid_dataset, test_dataset = piqa_to_dataframe(base_path=base_path)

    text_columns = ["goal", "sol1", "sol2"]

    for col in text_columns:
        train_dataset[f"{col}_tokens"] = train_dataset[col].apply(tokenizer.encode_as_ids)
        valid_dataset[f"{col}_tokens"] = valid_dataset[col].apply(tokenizer.encode_as_ids)
        test_dataset[f"{col}_tokens"] = test_dataset[col].apply(tokenizer.encode_as_ids)

    return train_dataset, valid_dataset, test_dataset


class PIQADataset(Dataset):
    '''
    Class that transform dataframe to dict, using Torch Built-in Dataset Base Class
    '''
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        return {
            "goal": row["goal_tokens"],
            "sol1": row["sol1_tokens"],
            "sol2": row["sol2_tokens"],
            "label": row["label"]
        }


# ================ Collator ========================

class PIQACollator:
    """
    Convert a list of PIQA samples into padded tensors.

    Output:
        goal_ids:
            LongTensor, shape (B, Lg)

        goal_mask:
            BoolTensor, shape (B, Lg)
            True = valid token
            False = padding

        sol_ids:
            LongTensor, shape (B, 2, Ls)

        sol_mask:
            BoolTensor, shape (B, 2, Ls)
            True = valid token
            False = padding

        labels:
            LongTensor, shape (B,)
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_id()

    def __call__(self, batch):

        goals = []
        solutions = []
        labels = []

        for item in batch:

            goals.append(
                torch.tensor(item["goal"], dtype=torch.long)
            )

            # 两个 candidate
            solutions.append([
                torch.tensor(item["sol1"], dtype=torch.long),
                torch.tensor(item["sol2"], dtype=torch.long)
            ])

            labels.append(item["label"])

        # =========================
        # Goal
        # =========================

        goal_ids = pad_sequence(
            goals,
            batch_first=True,
            padding_value=self.pad_id
        )

        goal_mask = goal_ids != self.pad_id

        # =========================
        # Solutions
        # =========================

        # 找到当前 batch 中所有 candidate 的最大长度
        max_sol_len = max(
            sol.numel()
            for sample in solutions
            for sol in sample
        )

        # 手动使用同一个 max_sol_len padding
        sol_ids = torch.full(
            (len(batch), 2, max_sol_len),
            self.pad_id,
            dtype=torch.long
        )

        for i, sample in enumerate(solutions):
            for j, sol in enumerate(sample):
                sol_ids[i, j, :len(sol)] = sol

        sol_mask = sol_ids != self.pad_id

        # =========================
        # Labels
        # =========================

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        return (
            goal_ids,
            goal_mask,
            sol_ids,
            sol_mask,
            labels
        )



def get_piqa_dataloaders(base_path:Path, tokenizer, batch_size:int):
    '''
    Entrance function for quickly get all required dataloaders
    '''
    train_dataset, valid_dataset, test_dataset = tokenize_piqa_dataframe(tokenizer=tokenizer, base_path=base_path)

    collator = PIQACollator(tokenizer)

    train_loader = DataLoader(
        PIQADataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )

    validate_loader = DataLoader(
        PIQADataset(valid_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )

    test_loader = DataLoader(
        PIQADataset(test_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )

    return train_loader, validate_loader, test_loader

