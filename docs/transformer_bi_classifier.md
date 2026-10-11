# Introduction

本文简要介绍 TransformerClassifier 的使用方法及其基本实现原理。

`TransformerClassifier` 是一个用于 PIQA 数据集的二分类器，可以接入标准的 PIQA Data Loader 数据进行训练。

# 实现原理

由于 PIQA 任务为简单的序列二分类任务，不涉及序列生成，因此不与要 Decoder，在架构上只需要 Encoder + 线性分类头 即可实现。关键组件：

- Endocer: 使用 Pytorch 提供的 **多头注意力** 组件作为一个个 Encoder 的层，整个 Encoder 包含若干个结构一致的多头注意层
- Linear Classifier：线性分类头将编码好的固定长度的序列映射为 Logits（目前暂定序列长度为 `Encoding_Dim`）

## Encoder

Encoder 接收已经编码好的完整 Token ID 序列。对于每个 Candidate，输入序列遵循：

```
[CLS] GOAL [SEP] SOLUTION [EOS]
```

Encoder 接收 Token ID 序列，在 `model.encode()` 中进行：

- Embedding：将 ID 映射为向量，并乘以 sqrt(Embedding_Dim) 进行缩放
- Positional Encoding：添加 Positional Embedding 信息；Positional Embedding 使用经典的 Sinusoidal Positional Encoding 算法，因此位置编码本身不包含可训练参数
- Transformer Encoder： 把 Embedding 送入一个个多头注意力层，进行语义更新和特征提取
- [CLS] Pooling：提取 [CLS] (index=0) 位置的聚合特征向量，作为整个 Goal–Solution 序列的固定长度语义表示

随后语义向量被送入 Linear Classifier 进行评分，返回评分的 Logits，后续可以用于计算交叉熵。

Sinusoidal Positional Encoding 的位置可视化图如下：

<img src="./figures/sinusoidal_positional_encoding.png" width="600">

由于 Goal 和 Solution 被放在同一个序列中，Self-Attention 可以直接建立 Goal Token 与 Solution Token 之间的关系：

```
[CLS] GOAL [SEP] SOLUTION [EOS]
          ↕       ↕
       Self-Attention
```

这是注意力提高模型表现的基本原理。

## Classification

对于 PIQA 中的每个样本，模型分别对两个 Candidate 进行完整序列编码。

两个 Candidate 的输入分别为：

```
Candidate 1:
[CLS] GOAL [SEP] SOLUTION 1 [EOS]

Candidate 2:
[CLS] GOAL [SEP] SOLUTION 2 [EOS]
```

两个 Candidate 使用同一个 Transformer Encoder 和 Linear Classifier，即模型参数在两个 Candidate 之间共享。

对于 Candidate 1：

```
[B, L1]
   ↓
Transformer Encoder
   ↓
[B, L1, D]
   ↓
取 [CLS]
   ↓
[B, D]
   ↓
Linear Classifier
   ↓
[B, 1]
```

对于 Candidate 2：

```
[B, L2]
   ↓
Transformer Encoder
   ↓
[B, L2, D]
   ↓
取 [CLS]
   ↓
[B, D]
   ↓
Linear Classifier
   ↓
[B, 1]
```

最后将两个 Candidate 的 Logit 拼接：

```
[B, 1] + [B, 1]
        ↓
[B, 2]
```

最终输出的 [B, 2] 表示两个 Candidate 的分类 Logits：

```
logits[:, 0] → Candidate 1
logits[:, 1] → Candidate 2
```

可以直接用于 CrossEntropyLoss：

```
loss = F.cross_entropy(
    logits,
    labels
)
```

其中 labels 的形状为 [B]，取值为 0 或 1。

# 初始化

```python
from scripts.models.transformer_bi_classifier import TransformerClassifier
import torch.optim as optim
import torch.nn.functional as F

tf_model = TransformerClassifier(vocab_size=tokenizer.vocab_size(),
                                 embedding_dim=params['embedding_dim'],
                                 padding_idx=tokenizer.pad_id(),
                                 max_seq_length=2048,
                                 num_heads=params['num_heads'],
                                 num_layers=params['num_layers'])

tf_optimiser = optim.SGD(params=tf_model.parameters(), lr=params['learning_rate'])
criterion = F.cross_entropy
```

各参数含义如下：

| 参数 | 来源 | 含义 |
|---|---|---|
| `vocab_size` | `tokenizer.vocab_size()` | Tokenizer 的词表大小，决定 Embedding 层可以接收多少种不同的 Token ID。 |
| `embedding_dim` | `params['embedding_dim']` | Token Embedding 的维度，同时也是 Transformer 的 `d_model`。决定每个 Token 使用多少维向量表示。 |
| `padding_idx` | `tokenizer.pad_id()` | Padding Token 的 ID。Embedding 层不会更新该位置对应的向量；同时用于识别并屏蔽 Padding Token。 |
| `max_seq_length` | `2048` | Sinusoidal Positional Encoding 支持的最大序列长度。当前设置为 2048，表示模型最多为前 2048 个位置生成位置编码。 |
| `num_heads` | `params['num_heads']` | Multi-Head Self-Attention 的注意力头数量。多个 head 可以从不同的表示子空间学习 Token 之间的关系。 |
| `num_layers` | `params['num_layers']` | Transformer Encoder Layer 的数量。每增加一层，都会对上一层产生的 Token 表示进行进一步的特征提取。 |

其中有一个重要约束：

```text
embedding_dim % num_heads == 0
```

因为 Multi-Head Attention 会将 `embedding_dim` 分割到不同的 attention heads 中，因此 Embedding Dimension **必须是头数的整数倍**。

# 训练

模型可以直接接入项目定义的标准数据进行训练：

```python
from time import perf_counter

tf_model.train()

epoch_loss = 0.0

for input_1, input_2, mask_1, mask_2, labels in train_loader:

    start_ts = perf_counter()
    # Forward
    tf_optimiser.zero_grad()

    logits = tf_model.forward(input_1, input_2, mask_1, mask_2)

    # Loss
    loss = criterion(
        input=logits,
        target=labels
    )

    # Backward
    loss.backward()

    tf_optimiser.step()

    # Record loss
    epoch_loss += loss.item()

    end_ts = perf_counter()
    time_elips = end_ts - start_ts
    
    print(
        f"time={time_elips:.2f}, "
        f"loss={loss.item():.2f}"
    )
```

由于 DataLoader 使用 Dynamic Padding，因此不同 Batch 的 L1 和 L2 可能不同。

注意：在自己的 PC 上进行训练时，不要将数据的 Batch Size 设置太高（推荐 200），否则**极易造成内存溢出**。

<img src="./figures/transformer_bs200_memory.png" width="600">

# Q & A

## 如何获得注意力矩阵

### 方法一：自己取出注意力投影矩阵，之后自己给 Embedding 进行一次手动计算注意力

```python
# 取出注意力投影矩阵
W_Q, W_K, W_V = attn.in_proj_weight.chunk(3, dim=0)

print(W_Q.shape)
print(W_K.shape)
print(W_V.shape)

# torch.Size([100, 100])
# torch.Size([100, 100])
# torch.Size([100, 100])
```

这种方法可以帮助理解 Transformer 内部的计算过程，但实现较为复杂，而且需要自行处理 Multi-Head 的拆分、缩放、Softmax、Mask 等逻辑。

### 方法二：利用原 Encoder，手动进行 Forward 计算

使用 `scripts/models/transformer_attention_tools` 中的工具函数 `extract_encoder_attentions` 进行题取。

PyTorch 的 nn.MultiheadAttention 本身支持返回 Attention Weights。由于 nn.TransformerEncoderLayer 默认不会直接向外暴露每一层的 Attention Weights，因此可以在不修改原模型结构的情况下，手动执行 Encoder Layer 的 Forward 计算，并提取各层、各个 Head 的注意力权重。

提取出的 Attention Weights 可用于后续的可视化分析，包括观察不同 Head 的注意力分布、比较不同 Encoder Layer 的信息交互模式，以及分析 [CLS] Token 对输入序列中其他 Token 的注意力分配。

Attention 权重提取流程：

1. 将输入 Token IDs 转换为 Embeddings，并根据模型配置添加 Positional Encoding。

2. 逐层执行原 Encoder 的 Self-Attention、残差连接、Layer Normalization 和 Feed-Forward Network，同时保存各层的 Attention Weights。

提取的 Attention Weights 形状为 **[batch_size, num_heads, sequence_length, sequence_length]**。其中，每个矩阵表示一个 Head 中各 Query Token 对各 Key Token 的注意力分配。

之后可以使用 `scripts/models/transformer_attention_tools` 中的工具函数 `plot_attention_heads` 直接绘制多头注意力。原理如下：

1. 使用 SentencePiece Tokenizer 将 Token IDs 映射为可读的 Token 标签。

2. 使用 Matplotlib 和 Seaborn 绘制 Attention Heatmap，并支持将不同 Heads 自动排列为多个子图。

3. 根据 Attention Mask 排除 Padding Token，并可通过样本索引选择较短的输入序列，以提高热图的可读性。

4. 将正确答案标签传入绘图函数，在热图标题中标注对应的候选方案，便于结合 PIQA 任务的真实标签分析模型行为。

需要注意的是，Attention Weights 反映的是模型的信息交互模式，并不直接等同于 Token 对最终分类结果的重要性。对于当前使用 [CLS] 表示进行分类的 Transformer，可以进一步观察最后一层 [CLS] 行的注意力分布，并结合预测结果或消融实验进行分析。