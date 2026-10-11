import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        max_length=512
    ):
        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(1)
        # [L, 1]

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        # [D/2]

        pe = torch.zeros(max_length, embedding_dim)
        # [L, D]

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # [L, D] -> [1, L, D]

        # 把 pe 保存为模型的一部分，但不要把它当成需要梯度更新的 parameter
        # 同时支持移动到 gpu
        self.register_buffer("pe", pe)


    def forward(self, x):
        """
        x:
            [B, L, D]
        """

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]



if __name__ == "__main__":
    # =========================
    # Test
    # =========================
    
    embedding_dim = 64
    max_length = 100

    pos_encoding = SinusoidalPositionalEncoding(
        embedding_dim=embedding_dim,
        max_length=max_length
    )

    pe = pos_encoding.pe

    print("PE shape:", pe.shape)
    print("PE min:", pe.min().item())
    print("PE max:", pe.max().item())


    # =========================
    # Visualisation
    # =========================
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    plt.imshow(
        pe.squeeze(0).numpy(),
        aspect="auto",
        cmap="coolwarm"
    )

    plt.colorbar(label="Encoding value")

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Positional Encoding")

    plt.tight_layout()
    plt.show()