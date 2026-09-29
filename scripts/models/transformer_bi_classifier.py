import torch
import torch.nn as nn
from scripts.models.sinusoidal_positional_encoding import SinusoidalPositionalEncoding


# 两分类 Transformer 不需要 Decoder，只需要一个线性分类头
# 最小实现，先不加 dropout 等优化曾

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=64,
        padding_idx = 3,
        max_seq_length = 2048,

        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        # 1. Embedding 层
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            # _freeze = True,   # 是否应该冻结？
            padding_idx=padding_idx
        )

        self.position_embedding = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_seq_length
        )

        # 2. 多头注意力 encoder
        # 对比实验：不同的 head 数对于精度的影响
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=128,
            batch_first=True
        )

        # 3. 多层 encoder，下一层是上一层的进一步抽象
        # 对比试验点：减少层数是否影响精度
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4. 线性分类头，映射成 logits
        # goal + sloution -> score
        self.classifier = nn.Linear(embedding_dim * 2, 1)


    def encode(self, ids, mask):
        """
        ids:
            [B, L]

        mask:
            [B, L]
            True  = valid token
            False = padding
        """

        x = self.embedding(ids)
        # [B, L] -> [B, L, D]

        x = self.position_embedding(x)
        # 加入 sinusoidal positional encoding
        # [B, L, D] -> [B, L, D]


        # Transformer expects:
        # src_key_padding_mask
        # True  = padding
        # False = valid token
        padding_mask = ~mask

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        # x shape: [B, L, D] -> 更新原始 token 的语义


        # Masked mean pooling
        # 对所有有效 token 的 embedding 求平均，得到一个固定长度的 goal / sol 语义表示
        # [B, L, D] -> [B, D]
        mask = mask.unsqueeze(-1)

        x = x.masked_fill(~mask, 0) # 把 padding (True) 位置全部填上 0

        lengths = mask.sum(dim=1).clamp(min=1)
        # 根据 mask 计算有多少个有效 token
        # clamp 防止在全都是 0 的情况下，length 变为 0，让下一步造成 ZeroDivision

        x = x.sum(dim=1) / lengths  # 有效 token / 有效长度
        # [B, L, D] -> [B, D]

        return x

    
    def forward(self, goal_ids, goal_mask, sol_ids, sol_mask):
        # -------------------------
        # Encode Goal
        # -------------------------

        goal_repr = self.encode(
            goal_ids,
            goal_mask
        )
        # [B, L, D] -> [B, D]

        # -------------------------
        # Encode Solution 1 & 2
        # Solution 需要把中间维度 2 合并到 Batch 中，再送入 Encoder
        # Encoding 完后，再展开到原始维度
        # -------------------------

        batch_size = sol_ids.shape[0]

        sol_ids = sol_ids.reshape(batch_size * 2, -1)
        # [B, 2, L] -> [2B, L]

        sol_mask = sol_mask.reshape(batch_size * 2, -1)
        # [B, 2, L] -> [2B, L]

        sol_repr = self.encode(sol_ids, sol_mask)
        # [2B, L] -> [2B, D]
        # 出来后，最后一维变为 embedding_size !!

        sol_repr = sol_repr.reshape(batch_size, 2, -1)
        # [2B, D] -> [B, 2, D]
        

        # -------------------------
        # Combine Goal + Solution
        # 语义信息已经被上面的 Encoder 更新完成，下面只负责线性映射
        # -------------------------

        goal_repr = goal_repr.unsqueeze(1)
        # [B, D] -> [B, 1, D]

        goal_repr = goal_repr.expand(-1, 2, -1)
        # 将同一个 Goal representation 复制到两个 candidate
        # [B, 1, D] -> [B, 2, D]

        features = torch.cat([goal_repr, sol_repr], dim=-1)
        # 把 goal 拼接到每个 solution 前面
        # [B, 2, 2D]

        # -------------------------
        # Score each candidate
        # -------------------------

        logits = self.classifier(features)
        # [B, 2, 2D] -> [B, 2, 1] 每个 goal + sol 的组合获得一个分类分数

        logits = logits.squeeze(-1)
        # [B, 2, 1] -> [B, 2] 将拆分的分数组合映射回两个选项

        return logits


