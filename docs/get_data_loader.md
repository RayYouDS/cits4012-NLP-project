# Introduction

PIQA 数据集已经封装为 PyTorch DataLoader，完成了数据读取、SentencePiece Tokenization、特殊 Token 添加以及动态 Padding。

模型可以直接使用 DataLoader 输出的数据，无需额外处理原始 PIQA 数据。

# Usage

```python
from piqa_dataloader import get_piqa_dataloaders
from pathlib import Path
import sentencepiece as spm

BASE_PATH = Path('../datasets/PIQA')

tokenizer = spm.SentencePieceProcessor(
    model_file="./tokenizer/piqa_bpe.model"
)

train_loader, validate_loader, test_loader = get_piqa_dataloaders(base_path=BASE_PATH, tokenizer=tokenizer, batch_size=1000)
```

调用 get_piqa_dataloaders() 后，会返回三个结构相同的 DataLoader 对象：

- `train_loader`: 训练数据
- `validate_loader`: 从训练数据中抽取的 10% 验证数据，用于训练过程中评估模型性能
- `test_loader`: 测试数据，仅用于最终测试，不参与模型训练或训练过程中的模型选择

每个对象都支持直接遍历：

```python
for input_1, input_2, mask_1, mask_2, labels in train_loader:
    print(input_1.shape, input_2.shape)
```

输出：

```
torch.Size([1000, 477]) torch.Size([1000, 477])
torch.Size([1000, 233]) torch.Size([1000, 231])
torch.Size([1000, 260]) torch.Size([1000, 260])
torch.Size([1000, 253]) torch.Size([1000, 253])
torch.Size([1000, 214]) torch.Size([1000, 209])
torch.Size([1000, 232]) torch.Size([1000, 233])
torch.Size([1000, 289]) torch.Size([1000, 289])
torch.Size([1000, 331]) torch.Size([1000, 329])
torch.Size([1000, 237]) torch.Size([1000, 237])
torch.Size([1000, 254]) torch.Size([1000, 256])
torch.Size([1000, 200]) torch.Size([1000, 202])
torch.Size([1000, 202]) torch.Size([1000, 203])
torch.Size([1000, 320]) torch.Size([1000, 322])
torch.Size([1000, 333]) torch.Size([1000, 333])
torch.Size([502, 174]) torch.Size([502, 174])
```

由于 DataLoader 使用 dynamic padding，每个 batch 的序列长度可能不同。

此外，input_1 和 input_2 的序列长度也可以不同。例如：

- input_1: torch.Size([1000, 233])
- input_2: torch.Size([1000, 231])

这是正常现象，因为两个 solution 的 Token 数量可能不同。

# DataLoader Output

每个 batch 返回以下五个 Tensor：

|Tensor|Shape|Description|
|--|--|--|
|`input_1`|`[B, L1]`|`[CLS] + goal + [SEP] + sol1 + [EOS]` 的 Token IDs|
|`input_2`|`[B, L2]`|`[CLS] + goal + [SEP] + sol2 + [EOS]` 的 Token IDs|
|`mask_1`|`[B, L1]`|`input_1` 的有效 Token Mask|
|`mask_2`|`[B, L2]`|`input_2` 的有效 Token Mask|
|`labels`|`[B]`|PIQA标签，0 表示 sol1 正确，1 表示 sol2 正确|


其中：

- `B` 为 batch size
- `L1`、`L2` 为当前 batch 中对应序列的最大长度
- `input_1` 和 `input_2` 的长度不要求相同

由于使用动态 Padding，短于当前 batch 最大长度的序列会使用 PAD Token 补齐。

# Input Sequence Format

每个 PIQA 样本会被转换为两个输入序列：

```
[CLS] goal [SEP] sol1 [EOS]
[CLS] goal [SEP] sol2 [EOS]
```

例如：

```
[CLS] To achieve a brown monkey color for the monkey cupcake [SEP]
Use chocolate frosting to ice the cupcake [EOS]

[CLS] To achieve a brown monkey color for the monkey cupcake [SEP]
Use green frosting to ice the cupcake [EOS]
```

模型需要比较两个输入序列，并根据 label 判断两个 solution 中哪个更加合理。

# Recovering Text from Token IDs

可以使用 SentencePiece Tokenizer 将 Token IDs 逆向还原为文本。

```python
input_1, input_2, mask_1, mask_2, labels = next(iter(train_loader))

text_1 = tokenizer.decode_ids(input_1[0][mask_1[0]].tolist())
text_2 = tokenizer.decode_ids(input_2[0][mask_2[0]].tolist())

print(text_1)
print(text_2)
```

输出：

```
[CLS] To achieve a brown monkey color for the monkey cupcake[SEP] Use chocolate frosting to ice the cupcake
[CLS] To achieve a brown monkey color for the monkey cupcake[SEP] Use green frosting to ice the cupcake
```

这里通过 mask 过滤掉了 Padding Token，因此只对实际有效的 Token 进行 Decode。

# Note

- mask_1 和 mask_2 用于区分有效 Token 和 Padding Token。对于 Transformer，Mask 可以用于 Attention 计算，以避免模型关注 Padding Token。对于 RNN、GRU、LSTM 等 Sequential 模型，Mask 可以用于计算每个序列的实际长度，随后可以结合 pack_padded_sequence 等机制减少 Padding 带来的无效计算。
- DataLoader 不会固定 Tokenizer 的实现细节，调用者需要提供一个已经初始化的 SentencePiece Tokenizer，如果后续更新了 Tokenizer，只需要加载新的模型文件并重新初始化 Tokenizer，而 DataLoader 的调用方式无需改变。