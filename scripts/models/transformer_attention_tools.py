import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math


@torch.no_grad()
def extract_encoder_attentions(model, ids, mask):
    '''
    用于取出 Attention Weight Matrix（注意力权重矩阵） 的工具函数

    Args:
        model: TransformerClassifier
        ids:  [B, L]
        mask: [B, L], True = valid token, False = padding

    Returns:
        attentions: list of tensors, each [B, H, L, L]
    '''
    # model.eval()  # 默认不进行状态管理，需要在函数外手动调整模型状态

    # Embedding + scaling
    x = model.embedding(ids) * math.sqrt(model.embedding_dim)

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

    # 自动获取当前样本的正确答案
    correct_candidate = None

    if labels is not None:
        label = labels[sample_idx].item()
        correct_candidate = f"input_{label + 1}"

    # 自动将 Token ID 转换为 SentencePiece Token
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