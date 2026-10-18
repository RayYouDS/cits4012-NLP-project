import math
import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, dimension):
        """
        Implements scaled dot-product self-attention.

        The computation is designed to align with the attention mechanism
        used in the Transformer, making comparisons between architectures
        more consistent in the experiments.
        """
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

        # output contains the context-enriched token representations,
        # while weights contains the attention weights used to compute them
        return output, weights