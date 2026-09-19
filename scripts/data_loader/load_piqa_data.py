from pathlib import Path
import pandas as pd

BASE_PATH = Path('./datasets/PIQA')


def piqa_to_dataframe(dir: Path, sample_prop: float=0.1) -> tuple:

    # ======== Make File Pathes ======================
    train_data_path = BASE_PATH / 'train.jsonl'
    train_label_path = BASE_PATH / 'train-labels.lst'
    test_data_path = BASE_PATH / 'test.jsonl'
    test_label_path = BASE_PATH / 'test-labels.lst'

    # ========== Load Dataset as DataFrames =========================
    train_dataset = pd.read_json(train_data_path, lines=True, encoding="utf-8")
    train_labels = pd.read_csv(train_label_path, header=None, names=["label"])

    train_dataset = pd.concat([train_dataset.reset_index(drop=True), train_labels], axis=1)

    test_dataset = pd.read_json(test_data_path, lines=True, encoding="utf-8")
    test_labels = pd.read_csv(test_label_path, header=None, names=["label"])

    test_dataset = pd.concat([test_dataset.reset_index(drop=True), test_labels], axis=1)

    # ==================== Sample (Train -> Train + Validation)
    sample_size = int(len(train_dataset) * sample_prop)

    valid_dataset = train_dataset.sample(n=sample_size, random_state=4012)
    train_dataset = train_dataset.drop(valid_dataset.index)


    return (train_dataset, valid_dataset, test_dataset)



train_dataset, valid_dataset, test_dataset = piqa_to_dataframe(dir=BASE_PATH)

print(train_dataset.shape)  # (14502, 4)
print(test_dataset.shape)   # (1838, 4)









