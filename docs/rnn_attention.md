# Introduction

本文简要介绍 `RNNWithAttention` 的使用方法及其基本实现原理。

`RNNWithAttention` 是一个用于 PIQA 数据集的二分类器，可以接入标准的 PIQA Data Loader 数据进行训练。

该模型支持三种 RNN 架构：

- `rnn`：PyTorch `nn.RNN`
- `gru`：PyTorch `nn.GRU`
- `lstm`：PyTorch `nn.LSTM`

模型默认使用 **Bidirectional RNN**，并在 RNN 输出的 Token 表示上进一步使用 Scaled Dot-Product Self-Attention。

# 实现原理

模型整体采用：

```
Embedding
    ↓
Bidirectional RNN / GRU / LSTM
    ↓
Self-Attention
    ↓
Masked Mean Pooling
    ↓
Linear Classifier
```

首先，Embedding 将 Token ID 转换为向量。

随后，RNN 从序列中提取上下文信息。模型默认使用 Bidirectional RNN，使每个 Token 都可以同时利用前后文信息。

RNN 输出之后，Self-Attention 用于进一步计算不同 Token 之间的相关性，使模型可以重新聚合序列中的重要信息。

最后，通过 Masked Mean Pooling 将整个序列压缩成一个固定长度的向量，并交给 Linear Classifier 对 Candidate 进行评分。

由于 PIQA 是一个二分类任务，不涉及序列生成，因此模型不需要 Decoder。

# 初始化

```python
from scripts.models.rnn_attention_refactored import RNNWithAttention
import torch.optim as optim
import torch.nn.functional as F

# 初始化模型和优化器
rnn_model = RNNWithAttention(vocab_size=tokenizer.vocab_size(),
                                 embedding_dim=params['embedding_dim'],
                                 hidden_dim=params['hidden_size'],
                                 padding_idx=tokenizer.pad_id(),
                                 architecture=params['architecture'],
                                 bidirectional=params['bidirectional'],
                                 enable_attention=params['enable_attention']
                                 )

rnn_optimiser = optim.Adam(params=rnn_model.parameters(), lr=params['learning_rate'])
criterion = F.cross_entropy
```

各参数含义如下：

| 参数 | 来源 | 含义 |
|---|---|---|
| `vocab_size` | `tokenizer.vocab_size()` | Tokenizer 的词表大小，决定 Embedding 层可以接收多少种不同的 Token ID。 |
| `embedding_dim` | `params['embedding_dim']` | Token Embedding 的维度，同时也是 RNN 的输入维度。 |
| `hidden_dim` | `params['hidden_size']` | RNN 每个方向的 Hidden State 维度。 |
| `padding_idx` | `tokenizer.pad_id()` | Padding Token 的 ID，用于 Embedding 层以及识别 Padding Token。 |
| `architecture` | `params['architecture']` | RNN 的具体架构，可以选择 `rnn`、`gru` 或 `lstm`。 |
| `bidirectional` | `params['bidirectional']` | 是否使用 Bidirectional RNN。开启后同时从正向和反向处理序列。 |
| `enable_attention` | `params['enable_attention']` | 是否启用 Self-Attention。可以用于 Attention Ablation Experiment。 |

其中需要注意 `hidden_dim` 与最终 RNN 输出维度的关系。

当使用 Bidirectional RNN 时：

```text
rnn_output_dim = hidden_dim × 2
```

例如：

```text
hidden_dim = 128
bidirectional = True

rnn_output_dim = 256
```

因此 Self-Attention 和 Linear Classifier 的输入维度都是：

```text
256
```

# 训练

模型可以直接接入项目定义的标准 Data Loader 进行训练：

```python
from time import perf_counter

rnn_model.train()

epoch_loss = 0.0

for input_1, input_2, mask_1, mask_2, labels in train_loader:

    start_ts = perf_counter()
    # Forward
    rnn_optimiser.zero_grad()

    logits, weights = rnn_model.forward(input_1, input_2, mask_1, mask_2)

    # Loss
    loss = criterion(
        input=logits,
        target=labels
    )

    # Backward
    loss.backward()

    rnn_optimiser.step()

    # Record loss
    epoch_loss += loss.item()

    end_ts = perf_counter()
    time_elips = end_ts - start_ts
    
    print(
        f"time={time_elips:.2f}, "
        f"loss={loss.item():.2f}"
    )

```

`forward()` 返回两个对象：

```python
logits, weights = rnn_model.forward(...)
```

其中：

```text
logits:
[B, 2]
```

表示两个 Candidate 的分类 Logits。

而：

```text
weights:
[weights_1, weights_2]
```

分别对应两个 Candidate 的 Self-Attention Weight：

```text
weights_1: [B, L1, L1]
weights_2: [B, L2, L2]
```

如果关闭 Attention 则：

```text
weights = [None, None]
```

由于 Data Loader 使用 Dynamic Padding，因此不同 Batch 的 `L1` 和 `L2` 可能不同。

# Attention Ablation

`enable_attention` 可以用于进行 Attention Ablation。

完整模型：

```text
Embedding
    ↓
BiGRU
    ↓
Self-Attention
    ↓
Masked Mean Pooling
    ↓
Classifier
```

关闭 Attention 后：

```text
Embedding
    ↓
BiGRU
    ↓
Last Hidden State
    ↓
Classifier
```

因此可以通过：

```python
enable_attention=True
```

和：

```python
enable_attention=False
```

比较模型性能，从而分析 Self-Attention 对模型表现的影响。

进行消融实验时，应保持以下条件一致：

- Training Dataset
- Validation Dataset
- Test Dataset
- Tokenizer
- Batch Size
- Optimiser
- Learning Rate
- Training Epochs
- Random Seed

只改变 `enable_attention`，以尽量保证实验结果能够归因于 Self-Attention。

# Prediction

模型提供 `predict()` 方法用于进行不计算梯度的 Forward：

```python
with torch.no_grad():
    logits, weights = rnn_model.predict(
        input_1,
        input_2,
        mask_1,
        mask_2
    )
```

`predict()` 使用 `torch.no_grad()`，因此不会构建用于反向传播的计算图。

需要注意：

```python
predict()
```

**不会自动调用 `model.eval()`**。

因此在进行验证、测试或性能评估时，应由调用方主动控制模型状态：

```python
rnn_model.eval()

with torch.no_grad():
    logits, weights = rnn_model.predict(
        input_1,
        input_2,
        mask_1,
        mask_2
    )
```

这种设计可以避免 `predict()` 在训练过程中意外修改模型的 `train/eval` 状态。

# Q & A

## RNN Attention 和 Transformer Attention 有什么区别？

两者都使用 Scaled Dot-Product Attention：

```text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
= Softmax(QKᵀ / √d) V
```

但两者的整体架构不同。

RNN 模型：

```text
Embedding
    ↓
BiGRU
    ↓
Self-Attention
    ↓
Pooling
```

Transformer：

```text
Embedding
    ↓
Positional Encoding
    ↓
Transformer Encoder
    ↓
CLS Pooling
```

当前 RNN 模型中的 Attention 是**单头 Self-Attention**：

```text
[B, L, D]
    ↓
Attention
    ↓
[B, L, D]

Attention Weights:
[B, L, L]
```

而 Transformer 使用 Multi-Head Attention：

```text
[B, L, D]
    ↓
Multi-Head Attention
    ↓
[B, L, D]

Attention Weights:
[B, H, L, L]
```

因此这里的 Attention 主要负责对 RNN 已经提取的上下文特征进行进一步的信息聚合。

## 为什么 RNN 使用 Bidirectional？

单向 RNN 只能按照一个方向处理序列。

例如：

```text
Token1 → Token2 → Token3 → Token4
```

Token 2 主要只能利用前面的上下文。

Bidirectional RNN 则同时进行：

```text
Forward:
Token1 → Token2 → Token3 → Token4

Backward:
Token4 → Token3 → Token2 → Token1
```

最后将两个方向的 Hidden State 拼接：

```text
Forward Hidden
      +
Backward Hidden
      ↓
Bidirectional Representation
```

因此每个 Token 都可以同时获得前后文信息。

对于 PIQA 的 Goal–Solution 判断任务，这种双向上下文表示可以用于进一步进行 Candidate 的语义表示。

## 为什么不能直接使用最后一个 RNN Hidden State？

当前模型在启用 Attention 时没有直接使用最后一个 Hidden State，而是：

```text
RNN Outputs
    ↓
Self-Attention
    ↓
Masked Mean Pooling
```

原因是最后一个 Hidden State 是一个相对压缩的序列表示，而 Attention 可以首先重新聚合整个序列的信息。

Masked Mean Pooling 随后对所有有效 Token 进行平均：

```text
Token 1 ─┐
Token 2 ─┤
Token 3 ─┼→ Mean → Sequence Representation
Token 4 ─┤
Padding ─┘
```

Padding Token 不参与计算。

在关闭 Attention 的 Ablation Model 中，则使用 RNN 的 Last Hidden State 作为 Sequence Representation。

## 如何获得 Attention Matrix？

`forward()` 本身已经返回 Attention Weights：

```python
logits, weights = rnn_model.forward(
    input_1,
    input_2,
    mask_1,
    mask_2
)
```

其中：

```python
weights[0]
```

对应 Candidate 1：

```text
[B, L1, L1]
```

而：

```python
weights[1]
```

对应 Candidate 2：

```text
[B, L2, L2]
```

例如取 Batch 中第一个样本：

```python
attention_1 = weights[0][0]
attention_2 = weights[1][0]
```

得到：

```text
attention_1: [L1, L1]
attention_2: [L2, L2]
```

可以进一步使用 Heatmap 对 Attention Matrix 进行可视化。

需要注意，由于 Data Loader 使用 Dynamic Padding，因此 `L1` 和 `L2` 不一定相同。

## 如何针对该模型进行消融实验？

当前最直接的 Ablation 是关闭 Self-Attention：

```text
Full Model:

BiGRU
  ↓
Self-Attention
  ↓
Masked Mean Pooling
  ↓
Classifier
```

对比：

```text
Ablation:

BiGRU
  ↓
Last Hidden State
  ↓
Classifier
```

实验时应保持其他条件一致，只改变 `enable_attention`。

此外，可以将：

```python
architecture="rnn"
architecture="gru"
architecture="lstm"
```

作为不同 RNN Architecture 的比较实验。

需要注意，这类实验更适合描述为 **Architecture Comparison**，而不是严格意义上的 Component Ablation，因为 RNN、GRU 和 LSTM 本身改变了整个序列建模组件。

**另外，应先进行序列模型与 Transformer 整体性能的对比，选出表现最好的模型后，再进一步对最好的模型进行实验。**
