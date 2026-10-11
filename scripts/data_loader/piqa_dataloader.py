from scripts.data_loader.load_piqa_data import piqa_to_dataframe
from pathlib import Path
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from scripts.data_loader.bucket_sampler import BucketBatchSampler


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


class PIQASequenceBuilder:
    '''
    Class that make input sequence: [CLS] goal [SEP] sol1/2 [EOS]
    '''
    def __init__(self, tokenizer, include_goal=True):
        self.cls_id = tokenizer.piece_to_id("[CLS]")
        self.sep_id = tokenizer.piece_to_id("[SEP]")
        self.eos_id = tokenizer.eos_id()
        self.include_goal = include_goal

        if self.cls_id == tokenizer.unk_id():
            raise ValueError("[CLS] is not defined in tokenizer")

        if self.sep_id == tokenizer.unk_id():
            raise ValueError("[SEP] is not defined in tokenizer")

    def build(self, goal, solution):

        if self.include_goal:
            return [
                self.cls_id,
                *goal,
                self.sep_id,
                *solution,
                self.eos_id
            ]

        else:
            return [
                self.cls_id,
                *solution,
                self.eos_id
            ]




class PIQACollator:
    def __init__(self, tokenizer, include_goal=True):

        self.pad_id = tokenizer.pad_id()

        self.sequence_builder = PIQASequenceBuilder(tokenizer, include_goal=include_goal)

    def __call__(self, batch):

        input_1 = []
        input_2 = []
        labels = []

        for item in batch:

            seq1 = self.sequence_builder.build(
                item["goal"],
                item["sol1"]
            )

            seq2 = self.sequence_builder.build(
                item["goal"],
                item["sol2"]
            )

            input_1.append(
                torch.tensor(seq1, dtype=torch.long)
            )

            input_2.append(
                torch.tensor(seq2, dtype=torch.long)
            )

            labels.append(item["label"])

        input_1 = pad_sequence(
            input_1,
            batch_first=True,
            padding_value=self.pad_id
        )

        input_2 = pad_sequence(
            input_2,
            batch_first=True,
            padding_value=self.pad_id
        )

        mask_1 = input_1 != self.pad_id
        mask_2 = input_2 != self.pad_id

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        return input_1, input_2, mask_1, mask_2, labels


def get_piqa_dataloaders(base_path:Path,
                         tokenizer, 
                         batch_size:int, 
                         include_goal: bool = True, 
                         use_bucket: bool = True, 
                         bucket_size_multiplier: int = 50):
    '''
    Entrance function for quickly get all required dataloaders
    '''
    train_dataset, valid_dataset, test_dataset = tokenize_piqa_dataframe(tokenizer=tokenizer, base_path=base_path)

    train_dataset = PIQADataset(train_dataset)
    valid_dataset = PIQADataset(valid_dataset)
    test_dataset = PIQADataset(test_dataset)

    # 让所有 Dataloader 使用相同的 collator 构造
    collator = PIQACollator(
        tokenizer,
        include_goal=include_goal
    )

    # 根据是否分桶，构造 Train dataloader （验证、测试仍然用旧方法不动它）
    if use_bucket:
        train_batch_sampler = BucketBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            include_goal=include_goal,
            bucket_size_multiplier=bucket_size_multiplier,
            shuffle=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collator,
        )
        
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )

    validate_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return train_loader, validate_loader, test_loader

