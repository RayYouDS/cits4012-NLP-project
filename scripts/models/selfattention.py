import math
import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, dimension):
        '''
        这个 Attention 使用 Scaled Dot-Product
        目的是尽可能与 Transformer 的 Attention 计算机制对齐
        使对照试验更有说服力
        '''
        super().__init__()
        self.dimension = dimension

        self.Wq = nn.Linear(dimension, dimension)
        self.Wk = nn.Linear(dimension, dimension)
        self.Wv = nn.Linear(dimension, dimension)

    def forward(self, x, padding_mask=None):
        # Q, K, V
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Attention score
        scores = Q @ K.transpose(-2, -1)

        # Scaled dot-product
        scores = scores / math.sqrt(self.dimension)

        # Padding mask
        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1),
                float("-inf")
            )

        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        output = weights @ V

        # output 是更新后的实际值（语义）
        # weight 是实际起作用的注意力权重矩阵
        return output, weights