import torch
import torch.nn as nn
from scripts.models.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
import math


# A binary classification Transformer does not require a decoder;
# an encoder and a linear classification head are sufficient.
# Dropout is used for regularisation.

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=64,
        enable_pos_emb = True,
        padding_idx = 3,
        max_seq_length = 2048,

        num_heads=4,
        num_layers=2,

        dropout_rate=0.1,
        pooling = "cls"
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            # Ensure that the embedding dimension is divisible by the number of heads
            raise ValueError("embedding_dim must be divisible by num_heads")

        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.pooling = pooling

        self.embedding_dim = embedding_dim
        self.enable_pos_emb = enable_pos_emb

        # 1. Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        if enable_pos_emb:
            self.position_embedding = SinusoidalPositionalEncoding(
                embedding_dim=embedding_dim,
                max_length=max_seq_length
            )

        # 1.5. Dropout after positional encoding, following the standard
        # Transformer design
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # 2. Multi-head attention encoder layer
        # Ablation study: investigate how the number of attention heads
        # affects classification accuracy
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=128,
            batch_first=True,
            dropout=dropout_rate  # Explicitly set dropout within the encoder layer
        )

        # 3. Stack multiple encoder layers, allowing successive layers
        # to learn increasingly contextualised representations
        # Ablation study: investigate the effect of reducing the number of layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3.5. Dropout before the classification head
        # This can be set to dropout_rate or a higher value, such as 0.2 or 0.3
        self.fc_dropout = nn.Dropout(p=dropout_rate)

        # 4. Linear classification head that produces a scalar logit per candidate
        self.classifier = nn.Linear(embedding_dim, 1)


    def encode(self, input_ids, mask):
        # 1. Convert token IDs into embeddings
        x = self.embedding(input_ids)

        if self.enable_pos_emb:
            # 2. Add sinusoidal positional encodings
            x = self.position_embedding(x)

        # Apply dropout to the embeddings, even when positional encoding is disabled
        x = self.emb_dropout(x)
        
        padding_mask = ~mask

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Pooling: [B, L, D] -> [B, D]
        if self.pooling == "cls":
            # Extract the representation at the [CLS] position (index 0).
            # Self-attention allows this position to incorporate contextual
            # information from the other tokens in the sequence.
            pooled = x[:, 0, :]


        elif self.pooling == "mean":
            # Compute the mean over valid token positions only, excluding padding
            valid_mask = mask.unsqueeze(-1).to(dtype=x.dtype)  # [B, L, 1]

            pooled = (x * valid_mask).sum(dim=1)               # [B, D]
            pooled = pooled / valid_mask.sum(dim=1).clamp(min=1)

        # Apply dropout before passing the representation to the classifier
        pooled = self.fc_dropout(pooled)
        
        return pooled

    
    def forward(
        self,
        input_1,
        input_2,
        mask_1,
        mask_2
    ):
        # Encode candidate 1 & 2
        repr_1 = self.encode(input_1, mask_1)
        repr_2 = self.encode(input_2, mask_2)

        # Compute a scalar logit for each candidate
        logit_1 = self.classifier(repr_1)   # [B, 1]
        logit_2 = self.classifier(repr_2)   # [B, 1]

        # [B, 1] + [B, 1] → [B, 2]
        logits = torch.cat(
            [logit_1, logit_2],
            dim=1
        )

        return logits

    @torch.no_grad()
    def predict(self, input_1, input_2, mask_1, mask_2):
        """
        Generate predictions without computing gradients.

        This method does not switch the model to evaluation mode automatically.
        This avoids unintentionally changing the model's training state.
        Set the model to eval() explicitly when performing evaluation
        or benchmarking.
        """
        outputs = self.forward(input_1, input_2, mask_1, mask_2)

        return outputs