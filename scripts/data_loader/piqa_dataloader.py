from load_piqa_data import piqa_to_dataframe
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


class PIQASequenceBuilder:
    '''
    Class that make input sequence: [CLS] goal [SEP] sol1/2 [EOS]
    '''
    def __init__(self, tokenizer):
        self.cls_id = tokenizer.piece_to_id("[CLS]")
        self.sep_id = tokenizer.piece_to_id("[SEP]")
        self.eos_id = tokenizer.eos_id()

        if self.cls_id == tokenizer.unk_id():
            raise ValueError("[CLS] is not defined in tokenizer")

        if self.sep_id == tokenizer.unk_id():
            raise ValueError("[SEP] is not defined in tokenizer")

    def build(self, goal, solution):

        return [
            self.cls_id,
            *goal,
            self.sep_id,
            *solution,
            self.eos_id
        ]





class PIQACollator:
    def __init__(self, tokenizer):

        self.pad_id = tokenizer.pad_id()

        self.sequence_builder = PIQASequenceBuilder(tokenizer)

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


def get_piqa_dataloaders(base_path:Path, tokenizer, batch_size:int):
    '''
    Entrance function for quickly get all required dataloaders
    '''
    train_dataset, valid_dataset, test_dataset = tokenize_piqa_dataframe(tokenizer=tokenizer, base_path=base_path)

    train_loader = DataLoader(
        PIQADataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PIQACollator(tokenizer)
    )

    validate_loader = DataLoader(
        PIQADataset(valid_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PIQACollator(tokenizer)
    )

    test_loader = DataLoader(
        PIQADataset(test_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PIQACollator(tokenizer)
    )

    return train_loader, validate_loader, test_loader



if __name__ == "__main__":
    '''
    # ============ Load PIQA DataFrames
    BASE_PATH = Path('./datasets/PIQA')
    train_dataset, valid_dataset, test_dataset = piqa_to_dataframe(base_path=BASE_PATH)


    # ============= Load Tokenizer ============================
    tokenizer = spm.SentencePieceProcessor(
        model_file="./scripts/tokenizer/piqa_bpe.model"
    )

    train_dataset, valid_dataset, test_dataset = tokenize_piqa_dataframe(tokenizer=tokenizer, base_path=BASE_PATH)

    print(train_dataset.columns)
    '''
    BASE_PATH = Path('./datasets/PIQA')
    tokenizer = spm.SentencePieceProcessor(
            model_file="./scripts/tokenizer/piqa_bpe.model"
    )
    train_loader, validate_loader, test_loader = get_piqa_dataloaders(base_path=BASE_PATH, tokenizer=tokenizer, batch_size=1000)