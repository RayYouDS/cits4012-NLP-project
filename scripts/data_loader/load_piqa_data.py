from pathlib import Path
import pandas as pd
import re

def clean_piqa_text(text: str) -> str:
    """清洗规则：仅进行规范化（Normalization），压缩多余连续空格，保留绝大部分原始语义、大小写与标点。"""
    if not isinstance(text, str):
        return text
    
    # 替换换行符、制表符等为空格
    text = re.sub(r"\s+", " ", text)
    # 去除首尾空格
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

    train_dataset = pd.concat([train_dataset.reset_index(drop=True), train_labels], axis=1)

    test_dataset = pd.read_json(str(test_data_path), lines=True, encoding="utf-8")
    test_labels = pd.read_csv(str(test_label_path), header=None, names=["label"])

    test_dataset = pd.concat([test_dataset.reset_index(drop=True), test_labels], axis=1)

    # New: Data Cleanse
    text_columns = [
        col
        for col in ["goal", "sol1", "sol2"]
        if col in train_dataset.columns
    ]

    for col in text_columns:
        train_dataset[col] = train_dataset[col].apply(clean_piqa_text)
        test_dataset[col] = test_dataset[col].apply(clean_piqa_text)

    # ==================== Sample (Train -> Train + Validation)
    sample_size = int(len(train_dataset) * sample_prop)

    valid_dataset = train_dataset.sample(n=sample_size, random_state=4012)
    train_dataset = train_dataset.drop(valid_dataset.index)


    return (train_dataset, valid_dataset, test_dataset)


if __name__ == "__main__":
    BASE_PATH = Path('./datasets/PIQA')

    train_dataset, valid_dataset, test_dataset = piqa_to_dataframe(BASE_PATH)

    print(train_dataset.shape)  # (14502, 4)
    print(test_dataset.shape)   # (1838, 4)









