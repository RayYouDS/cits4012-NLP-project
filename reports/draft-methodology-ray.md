# 2 Methodology

2.1 Task Formulation

The PIQA task is formulated as a binary classification problem. Given a goal ($g$) and two candidate solutions ($s_1$ and $s_2$), the model predicts which solution is more physically plausible. Each candidate is processed as a separate input sequence, constructed as $[CLS] g [SEP] s_j [EOS]$, where $j \in {1,2}$. The target label $y \in {0,1}$ indicates whether the first or second candidate is correct, respectively. All model parameters are randomly initialised and trained from scratch without pretrained language models.

2.2 Architecture Selection

We compare attention-based BiGRU, attention-based BiLSTM, and Transformer encoders under a common task formulation and evaluation procedure. All three achieved similar preliminary accuracies of approximately 60%. The Transformer was selected for subsequent experiments because its observed training behaviour showed a slower increase in the overfitting gap, rather than a clear initial accuracy advantage.

Figure~\ref{fig} illustrates the recurrent and Transformer encoder designs. Both use the same candidate-scoring framework, in which each candidate is encoded independently and scored by a shared linear layer.

2.3 Transformer Architecture

The Transformer classifier comprises trainable token embeddings, optional sinusoidal positional encodings, stacked encoder layers, pooling, and a linear scoring head. Its default configuration uses an embedding dimension of 64, four attention heads, two encoder layers, a feed-forward dimension of 128, and dropout of 0.1. Positional encodings are fixed, non-trainable buffers added to token embeddings.

Each encoder layer applies multi-head self-attention and a position-wise feed-forward network, with residual connections, layer normalisation, and dropout. Scaled dot-product attention is defined as

$$
\operatorname{softmax}\left(
\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_k}}+\mathbf M
\right)\mathbf V,
$$

where (d_k) is the key dimension and (\mathbf M) masks padded key positions. Transformer layers use multiple attention heads to learn contextual representations. The encoded sequence is reduced to a fixed-dimensional vector using either the first token (([\mathrm{CLS}])) or masked mean pooling; the default is ([\mathrm{CLS}]) pooling. Padding masks prevent padded tokens from being treated as valid attention keys \cite{vaswani2017attention}.

2.4 Recurrent Baselines and Shared Candidate Scoring

The BiGRU and BiLSTM baselines use trainable embeddings followed by bidirectional recurrent encoders. Their forward and backward hidden states are concatenated to form contextual token representations. The default embedding dimension is 64, with 128 hidden units per direction, yielding a 256-dimensional representation. Sequences are packed according to their valid lengths to avoid recurrent computation over padding tokens.

A single-head scaled dot-product self-attention module operates on the recurrent representations, followed by masked mean pooling. Attention-based recurrent architectures have previously been applied to text classification, although their effectiveness depends on the task and design \cite{yang2016hierarchical}. In our implementation, attention weights mask padded key positions, and the pooled representation is computed as

[
\mathbf r =
\frac{\sum_{t=1}^{L}m_t\widetilde{\mathbf h}t}
{\max(1,\sum{t=1}^{L}m_t)},
]

where (m_t=1) for valid tokens and 0 for padding. When attention is disabled, the model instead concatenates the final forward and backward hidden states. This ablation therefore compares the overall representation and pooling designs, rather than isolating attention alone.

For all three architectures, a shared linear layer scores each candidate representation (\mathbf r_j):

[
z_j=\mathbf w^\top\mathbf r_j+b,\qquad j\in{1,2}.
]

The two scores form the logits for binary classification, with class 0 corresponding to the first candidate and class 1 to the second. Models are trained using cross-entropy loss. Unlike recurrent attention, which operates on representations already contextualised by bidirectional recurrence, Transformer self-attention contextualises token representations within each encoder layer.

2.5 Design Rationale and Limitations

The shared scoring framework provides a consistent comparison across architectures without separate scoring parameters for the two candidate positions. However, candidates are encoded independently, without explicit cross-candidate interaction. The similar preliminary accuracies also mean that the Transformer showed no clear initial performance advantage; its selection was partly motivated by its observed training behaviour. Finally, attention weights describe how representations are combined but do not necessarily provide faithful explanations of predictions or establish genuine physical understanding.