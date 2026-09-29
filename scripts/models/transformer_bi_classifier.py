import torch
import torch.nn as nn
from scripts.models.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
import math


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

        self.embedding_dim = embedding_dim

        # 1. Embedding 层
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
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
        self.classifier = nn.Linear(embedding_dim, 1)


    def encode(self, input_ids, mask):
        # 1. 获取 Embedding 并乘以 sqrt(d_model) 保持数值量级匹配
        x = self.embedding(input_ids)
        x = self.embedding(input_ids) * math.sqrt(self.embedding_dim)

        # 2. 叠加正弦位置编码
        x = self.position_embedding(x)

        padding_mask = ~mask

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # 提取 [CLS] (index=0) 位置的聚合特征向量 -> [B, D]
        return x[:, 0, :]

    
    def forward(
        self,
        input_1,
        input_2,
        mask_1,
        mask_2
    ):

        repr_1 = self.encode(input_1, mask_1)
        repr_2 = self.encode(input_2, mask_2)

        logit_1 = self.classifier(repr_1)   # [B, 1]
        logit_2 = self.classifier(repr_2)   # [B, 1]

        # [B, 1] + [B, 1] → [B, 2]
        logits = torch.cat(
            [logit_1, logit_2],
            dim=1
        )

        return logits
