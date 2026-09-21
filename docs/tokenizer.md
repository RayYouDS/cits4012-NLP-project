# Intro

本项目使用预训练的 SentencePiece Tokenizer 对 PIQA 文本进行 Tokenization。

使用 Tokenizer 前，需要先加载预训练的 .model 文件：

```python
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor(
    model_file="./scripts/tokenizer/piqa_bpe.model"
)
```
加载完成后，tokenizer 对象可以用于：

- 文本 → Token / Token ID
- Token ID → 文本
- 查询 Special Tokens
- 查询 Vocabulary Size

# Tokenizer Files

保存的 Tokenizer 包含以下两个文件：

- `piqa_bpe.model` - Tokenizer 的模型文件，包含运行 Tokenization 所需的信息
- `piqa_bpe.vocab` - Vocabulary 的可读文本表示

加载时只需要加载 `.model` 模型本体即可。

# Usage Examples

## Get Special Tokens

可以通过 SentencePiece API 获取预定义的 Special Token ID：

```python
print("UNK:", tokenizer.unk_id())  # 0
print("BOS:", tokenizer.bos_id())  # 1
print("EOS:", tokenizer.eos_id())  # 2
print("PAD:", tokenizer.pad_id())  # 3
```

本项目另外定义了 [CLS] 和 [SEP] 两个 Special Tokens，可以通过 piece_to_id() 获取其 Token ID：

```python
cls_id = tokenizer.piece_to_id("[CLS]")
sep_id = tokenizer.piece_to_id("[SEP]")

print(cls_id, sep_id)   # 4 5
```

所有的 Special Tokens ID 应由 Tokenizer 查询获得，而不建议在模型代码中直接硬编码。

## Sentence to Tokens

Tokenizer 可以直接将文本转换为 Token Pieces 和 Token IDs：

```python
text = "An unseen-word, café, or 😊 is still tokenizable."

pieces = tokenizer.encode(text, out_type=str)
ids = tokenizer.encode(text, out_type=int)

print(pieces)
print(ids)
```

输出：

```
['▁An', '▁un', 'se', 'en', '-', 'word', ',', '▁c', 'af', 'é', ',', '▁or', '▁', '<0xF0>', '<0x9F>', '<0x98>', '<0x8A>', '▁is', '▁still', '▁to', 'k', 'en', 'iz', 'able', '.']
[3460, 779, 319, 299, 7932, 6122, 7915, 269, 1390, 7976, 7915, 366, 7892, 246, 165, 158, 144, 379, 1726, 278, 7914, 299, 868, 774, 7913]
```

模型的 Embedding 层应使用 Token IDs，而不是字符串 Token。

## Tokens to Sentence

可以使用 Token IDs 将 Token 序列还原为原始文本：

```python
restored = tokenizer.decode(ids)
print(restored)
```

输出：

```
An unseen-word, café, or 😊 is still tokenizable.
```

如果已经获得的是 Token Pieces，也可以通过 decode_pieces() 进行还原：

```python
restored = tokenizer.decode_pieces(pieces)
print(restored)
```

## Get Vocabulary Size

可以通过 get_piece_size() 获取 Vocabulary Size：

```python
vocab_size = tokenizer.get_piece_size()
print(vocab_size)
```

```
8000
```

Vocabulary Size 通常用于确定模型 Embedding 层的输入维度。

# Note

Vocabulary Size、Tokenization 参数等配置后续可能根据实验需要进行调整。如果 Tokenizer 发生变化，需要重新检查 Vocabulary Size 以及 Special Token IDs，并相应调整模型配置。

