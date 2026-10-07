from pathlib import Path
import pandas as pd
import re
import numpy as np
import unicodedata

def clean_piqa_text(text: str) -> str:
    """清洗规则：仅进行规范化（Normalization），压缩多余连续空格，保留绝大部分原始语义、大小写与标点。"""
    if not isinstance(text, str):
        return text

    # 1. 将 \xa0 (NBSP) 等特殊 Unicode 空白转化为标准 ASCII 空格 (\x20)
    text = unicodedata.normalize("NFKC", text)

    # 2. 替换换行符、制表符等为空格
    text = re.sub(r"\s+", " ", text)
    
    # 3. 去除首尾空格
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


    # ========== Group-Based Splitting (按 Goal 切分规避泄露) ===========
    # 1. 提取所有不重复的 Goal 列表
    unique_goals = full_train_df['goal'].unique()

    # 2. 对唯一 Goal 进行固定随机种子的 Shuffle
    rng = np.random.default_rng(seed=4012)
    rng.shuffle(unique_goals)

    # 3. 按比例切分 Goal 集合（确保同一个 Goal 的所有行严格落在同侧）
    num_val_goals = int(len(unique_goals) * sample_prop)
    val_goals = set(unique_goals[:num_val_goals])

    # 4. 根据 Goal 映射提取行索引（布尔掩码）
    val_mask = full_train_df['goal'].isin(val_goals)

    # 通过布尔掩码划分训练集和测试即
    valid_dataset = full_train_df[val_mask].reset_index(drop=True)
    train_dataset = full_train_df[~val_mask].reset_index(drop=True)


    return (train_dataset, valid_dataset, test_dataset)


if __name__ == "__main__":
    BASE_PATH = Path('./datasets/PIQA')

    train_dataset, valid_dataset, test_dataset = piqa_to_dataframe(BASE_PATH)

    print(train_dataset.shape)  # (14502, 4)
    print(test_dataset.shape)   # (1838, 4)









