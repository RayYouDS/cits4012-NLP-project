from scripts.data_loader.load_piqa_data import piqa_to_dataframe
from pathlib import Path
import random


BASE_PATH = Path('./datasets/PIQA')
train_dataset, valid_dataset, test_dataset = piqa_to_dataframe(base_path=BASE_PATH)

# ================= Prepare temporary training corpus for Tokenizer ===========================
temp_corpus = []
temp_corpus.extend(train_dataset['goal'].to_list())
temp_corpus.extend(train_dataset['sol1'].to_list())
temp_corpus.extend(train_dataset['sol2'].to_list())

random.Random(4012).shuffle(temp_corpus)


# =============== Save temp corpus ================================
output_path = Path("./scripts/tokenizer")

temp_corpus_path =  output_path / 'temp_piqa_corpus.txt'
with temp_corpus_path.open("w", encoding="utf-8") as f:
    for text in temp_corpus:
        # clean potential returns inside corpus
        text = str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        f.write(text + "\n")


# ======================= Train Tokenizer =====================================
import sentencepiece as spm

model_prefix = output_path / 'test_bpe'

spm.SentencePieceTrainer.train(input=str(temp_corpus_path),
                                model_prefix=str(model_prefix),
                                model_type="bpe",

                                vocab_size=8000,
                                character_coverage=1.0,
                                byte_fallback=True,

                                unk_id=0,
                                bos_id=1,
                                eos_id=2,
                                pad_id=3,

                                user_defined_symbols=["[CLS]", "[SEP]"]
                            )
