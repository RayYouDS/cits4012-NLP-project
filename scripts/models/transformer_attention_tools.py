import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math


@torch.no_grad()
def extract_encoder_attentions(model, ids, mask):
    """
    Extract attention weight matrices from a Transformer encoder.

    Args:
        model: A TransformerClassifier instance.
        ids: Token IDs with shape [B, L].
        mask: Boolean mask with shape [B, L], where True indicates
              a valid token and False indicates padding.

    Returns:
        A list of attention weight tensors, one per encoder layer.
        Each tensor has shape [B, H, L, L], where:
            B: batch size
            H: number of attention heads
            L: sequence length
    """
    # model.eval() is intentionally not called here.
    # The caller is responsible for managing the model's training state.

    # Embedding (no scaling)
    x = model.embedding(ids)

    # Optional positional encoding
    if model.enable_pos_emb:
        x = model.position_embedding(x)

    x = model.emb_dropout(x)
    padding_mask = ~mask

    attentions = []

    for layer in model.encoder.layers:

        # Self-Attention
        attn_output, attn_weights = layer.self_attn(
            x, x, x,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        attentions.append(attn_weights.detach())

        # Residual + LayerNorm
        x = layer.norm1(x + layer.dropout1(attn_output))

        # Feed-Forward
        ff_output = layer.linear2(
            layer.dropout(
                layer.activation(layer.linear1(x))
            )
        )

        # Residual + LayerNorm
        x = layer.norm2(x + layer.dropout2(ff_output))

    # Apply optional final encoder normalization
    if model.encoder.norm is not None:
        x = model.encoder.norm(x)

    return attentions



def plot_attention_heads(
    attentions,
    layer_idx=0,
    sample_idx=0,
    input_ids=None,
    tokenizer=None,
    mask=None,
    figsize_per_head=(5, 4),
    ncols=2,
    labels=None
):
    sample_attention = attentions[layer_idx][sample_idx]
    num_heads = sample_attention.shape[0]
    nrows = math.ceil(num_heads / ncols)

    # Identify the correct candidate, if the ground-truth label is provided
    correct_candidate = None

    if labels is not None:
        label = labels[sample_idx].item()
        correct_candidate = f"input_{label + 1}"

    # Convert token IDs into SentencePiece token strings
    tokens = None
    if input_ids is not None and tokenizer is not None:
        ids = input_ids[sample_idx].detach().cpu().tolist()
        tokens = [tokenizer.IdToPiece(token_id) for token_id in ids]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_head[0] * ncols,
            figsize_per_head[1] * nrows,
        ),
        squeeze=False,
    )
    axes = axes.flatten()

    valid = None
    if mask is not None:
        valid = mask[sample_idx].bool().cpu().numpy()

        if tokens is not None:
            tokens = [
                token for token, is_valid in zip(tokens, valid)
                if is_valid
            ]

    for head_idx in range(num_heads):
        attention = sample_attention[head_idx].detach().cpu().numpy()

        if valid is not None:
            attention = attention[valid][:, valid]

        sns.heatmap(
            attention,
            ax=axes[head_idx],
            xticklabels=tokens if tokens is not None else "auto",
            yticklabels=tokens if tokens is not None else "auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        axes[head_idx].set_title(f"Head {head_idx + 1}")
        axes[head_idx].set_xlabel("Key Token")
        axes[head_idx].set_ylabel("Query Token")

    for idx in range(num_heads, len(axes)):
        fig.delaxes(axes[idx])

    title = f"Layer {layer_idx + 1}: Attention Heads"

    if correct_candidate is not None:
        title += f"\nCorrect Answer: {correct_candidate}"

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()