from pathlib import Path
import pandas as pd
import re
import numpy as np
import unicodedata

def clean_piqa_text(text: str) -> str:
    """
    Clean text using Unicode normalisation and whitespace standardisation.

    This preserves the original casing, punctuation, and most semantic content.
    """
    if not isinstance(text, str):
        return text

    # 1. Convert special Unicode characters, such as non-breaking spaces (NBSP),
    # into their standard equivalents using NFKC normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Replace consecutive whitespace characters, including newlines
    # and tabs, with a single space
    text = re.sub(r"\s+", " ", text)
    
    # 3. Remove leading and trailing whitespace
    return text.strip()


def piqa_to_dataframe(base_path: Path, sample_prop: float=0.1) -> tuple:

    # ======== Make File Pathes ======================
    train_data_path = base_path / 'train.jsonl'
    train_label_path = base_path / 'train-labels.lst'
    test_data_path = base_path / 'test.jsonl'
    test_label_path = base_path / 'test-labels.lst'

    # Validate File Existence
    required_files = [
        train_data_path,
        train_label_path,
        test_data_path,
        test_label_path
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # ========== Load Dataset as DataFrames =========================
    train_dataset = pd.read_json(str(train_data_path), lines=True, encoding="utf-8")
    train_labels = pd.read_csv(str(train_label_path), header=None, names=["label"])
    full_train_df = pd.concat([train_dataset.reset_index(drop=True), train_labels], axis=1)

    test_dataset = pd.read_json(str(test_data_path), lines=True, encoding="utf-8")
    test_labels = pd.read_csv(str(test_label_path), header=None, names=["label"])
    test_dataset = pd.concat([test_dataset.reset_index(drop=True), test_labels], axis=1)

    # New: Data Cleanse
    text_columns = [col for col in ["goal", "sol1", "sol2"] if col in full_train_df.columns]

    for col in text_columns:
        full_train_df[col] = full_train_df[col].apply(clean_piqa_text)
        test_dataset[col] = test_dataset[col].apply(clean_piqa_text)


    # ========== Group-Based Train/Validation Split ====================
    # Split by unique goals to prevent identical goals from appearing
    # in both the training and validation sets

    # 1. Extract all unique goals
    unique_goals = full_train_df['goal'].unique()

    # 2. Shuffle the unique goals using a fixed random seed
    rng = np.random.default_rng(seed=4012)
    rng.shuffle(unique_goals)

    # 3. Select the specified proportion of unique goals for validation
    # All rows sharing the same goal will be assigned to the same split
    num_val_goals = int(len(unique_goals) * sample_prop)
    val_goals = set(unique_goals[:num_val_goals])

    # 4. Create a Boolean mask to identify validation rows
    val_mask = full_train_df['goal'].isin(val_goals)

    # 5. Construct the training and validation DataFrames
    valid_dataset = full_train_df[val_mask].reset_index(drop=True)
    train_dataset = full_train_df[~val_mask].reset_index(drop=True)


    return (train_dataset, valid_dataset, test_dataset)


if __name__ == "__main__":
    BASE_PATH = Path('./datasets/PIQA')

    train_dataset, valid_dataset, test_dataset = piqa_to_dataframe(BASE_PATH)

    print(train_dataset.shape)  # (14502, 4)
    print(test_dataset.shape)   # (1838, 4)









