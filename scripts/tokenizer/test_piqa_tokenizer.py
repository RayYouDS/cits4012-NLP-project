# Load tokenizer and test
import sentencepiece as spm

# ============= Load Tokenizer ============================
tokenizer = spm.SentencePieceProcessor(
    model_file="./scripts/tokenizer/piqa_bpe.model"
)

text = "An unseen-word, café, or 😊 is still tokenizable."

# ============== Sentence -> pieces and ids ====================
pieces = tokenizer.encode(text, out_type=str)
ids = tokenizer.encode(text, out_type=int)

print(pieces)
print(ids)

# ================ Sentence Restore ==========================
# restore sentence
restored = tokenizer.decode(ids)
print(restored)

# =============== Check Vocab Size ========================
vocab_size = tokenizer.get_piece_size()
print(vocab_size)


# ================ Test Special Tokens ========================
print("UNK:", tokenizer.unk_id())  # 0
print("BOS:", tokenizer.bos_id())  # 1
print("EOS:", tokenizer.eos_id())  # 2
print("PAD:", tokenizer.pad_id())  # 3

cls_id = tokenizer.piece_to_id("[CLS]")
sep_id = tokenizer.piece_to_id("[SEP]")

print(cls_id, sep_id)