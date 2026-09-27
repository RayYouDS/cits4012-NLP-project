# Introduction

PIQA 数据集已经封装为 PyTorch DataLoader，完成了数据读取、SentencePiece Tokenization、以及动态 Padding。

模型可以直接使用 DataLoader 输出的数据，无需额外处理原始 PIQA 数据。

# Usage

```python
from scripts.data_loader.piqa_dataloader import get_piqa_dataloaders
from pathlib import Path
import sentencepiece as spm

BASE_PATH = Path('./datasets/PIQA')

tokenizer = spm.SentencePieceProcessor(
    model_file="./scripts/tokenizer/piqa_bpe.model"
)

train_loader, validate_loader, test_loader = get_piqa_dataloaders(base_path=BASE_PATH, tokenizer=tokenizer, batch_size=1000)
```

调用 get_piqa_dataloaders() 后，会返回三个结构相同的 DataLoader 对象：

- `train_loader`: 训练数据
- `validate_loader`: 从训练数据中抽取的 10% 验证数据，用于训练过程中评估模型性能
- `test_loader`: 测试数据，仅用于最终测试，不参与模型训练或训练过程中的模型选择

每个对象都支持直接遍历：

```python
for goal_ids, goal_mask, sol_ids, sol_mask, labels in train_loader:
    print(goal_ids.shape, sol_ids.shape)
```

输出：

```
torch.Size([1000, 27]) torch.Size([1000, 2, 310])
torch.Size([1000, 29]) torch.Size([1000, 2, 470])
torch.Size([1000, 31]) torch.Size([1000, 2, 244])
torch.Size([1000, 26]) torch.Size([1000, 2, 208])
torch.Size([1000, 31]) torch.Size([1000, 2, 280])
torch.Size([1000, 35]) torch.Size([1000, 2, 195])
torch.Size([1000, 25]) torch.Size([1000, 2, 252])
torch.Size([1000, 24]) torch.Size([1000, 2, 187])
torch.Size([1000, 27]) torch.Size([1000, 2, 180])
torch.Size([1000, 26]) torch.Size([1000, 2, 321])
torch.Size([1000, 33]) torch.Size([1000, 2, 157])
torch.Size([1000, 36]) torch.Size([1000, 2, 242])
torch.Size([1000, 29]) torch.Size([1000, 2, 226])
torch.Size([1000, 31]) torch.Size([1000, 2, 326])
torch.Size([502, 24]) torch.Size([502, 2, 205])
```

由于 DataLoader 使用 dynamic padding，每个 batch 的序列长度（Lg, Ls）可能不同。

sol_ids 的 shape 为 `torch.Size([B, 2, Ls])`，代表两个 Solution 经过 Padding 后的等长 ID

# DataLoader Output

每个 batch 返回以下五个 Tensor：

|Tensor|Shape|Description|
|--|--|--|
|`goal_ids`|`[B, Lg]`| Goal 的 Token IDs |
|`goal_mask`|`[B, Lg]`| Goal 的有效 Token Mask |
|`sol_ids`|`[B, 2, Ls]`| 每个 Goal 对应的两个 Solution 的 Token IDs |
|`sol_mask`|`[B, 2, Ls]`| Solution IDs 的有效 Token Mask |
|`labels`|`[B]`| PIQA标签，0 表示 sol1 正确，1 表示 sol2 正确|


其中：

- `B` - batch size
- `Lg` - Goal Token 序列的长度
- `Ls` - Solution 序列的最大长度

由于使用动态 Padding，短于 Solution 序列的最大长度的 另一个 Solution 序列会使用 PAD Token 补齐。

# Input Sequence Format

每个 PIQA 样本会被转换为两个输入序列：

```
[goal tokens]
[[solution 1 tokens],
 [solution 2 tokens]]
```

例如：

```
Goal: [To get a strike in bowling,]
Solutions: [[knock down 10 pins with a single bowl.],
            [knock down 8 pins with a single bowl.]]
```

模型需要接收 Goal，并比较两个 Solution 序列，判断两个 solution 中哪个更加合理，最后输出一个长度为 [2] 的 logits 序列。

# Recovering Text from Token IDs

可以使用 SentencePiece Tokenizer 将 Token IDs 逆向还原为文本。

```python
goal_ids, goal_mask, sol_ids, sol_mask, labels = next(iter(train_loader))

goal_text = tokenizer.decode_ids(goal_ids[0][goal_mask[0]].tolist())
sol_1_text = tokenizer.decode_ids(sol_ids[0][0][sol_mask[0][0]].tolist())
sol_2_text = tokenizer.decode_ids(sol_ids[0][1][sol_mask[0][1]].tolist())

print(goal_text)
print(sol_1_text)
print(sol_2_text)
```

输出：

```
how do you poke something?
touch it with three fingers.
touch it with one finger.
```

注：由于使用了 Mask 和 Goal, Solution 语义分离，因此不再需要 [EOS], [SEP] 等特殊标记 Token。

# Note

- goal_mask 和 sol_mask 用于区分有效 Token 和 Padding Token。对于 Transformer，Mask 可以用于 Attention 计算，以避免模型关注 Padding Token。对于 RNN、GRU、LSTM 等 Sequential 模型，Mask 可以用于计算每个序列的实际长度，随后可以结合 pack_padded_sequence 等机制减少 Padding 带来的无效计算。
- DataLoader 不会固定 Tokenizer 的实现细节，调用者需要提供一个已经初始化的 SentencePiece Tokenizer，如果后续更新了 Tokenizer，只需要加载新的模型文件并重新初始化 Tokenizer，而 DataLoader 的调用方式无需改变。