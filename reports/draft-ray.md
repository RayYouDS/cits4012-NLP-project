正文不可以超过 6 页，reference 和 contribution 不算如正文

项目没有指定引用风格的情况下，IEEE + IEEEtran 更加 Latex 友好，配置直接，适合计算机科学报告。

# Title

# 1 Introduction

Physical commonsense reasoning is an important challenge in natural language processing (NLP), requiring models to reason about everyday physical interactions rather than relying solely on linguistic patterns. Bisk et al. introduced the Physical Interaction: Question Answering (PIQA) benchmark to evaluate this capability [1]. Given a goal and two candidate solutions, a model must select the solution that is more physically plausible. The task is challenging because plausible solutions often depend on implicit knowledge of objects, materials, and physical interactions that may not be explicitly described in text.

This project develops and evaluates neural models for PIQA, formulated as a binary classification problem. We compare attention-based bidirectional LSTM (BiLSTM) and bidirectional GRU (BiGRU) models with a Transformer-based model, and investigate the contribution of attention mechanisms to classification performance. Using a consistent evaluation pipeline, we assess classification accuracy and observed training behaviour to compare these architectures.

Our results show that [experimental findings]. Further experiments indicate that [key findings], highlighting [main implication]. These findings provide an empirical comparison of neural approaches to physical commonsense question answering and illustrate the challenges of learning physical knowledge from language data.

# 2 Methodology

2.1 Task Formulation

The PIQA task is formulated as a binary classification problem. Given a goal ($g$) and two candidate solutions ($s_1$ and $s_2$), the model predicts which solution is more physically plausible. Each candidate is processed as a separate input sequence, constructed as $[CLS] g [SEP] s_j [EOS]$, where $j \in {1,2}$. The target label $y \in {0,1}$ indicates whether the first or second candidate is correct, respectively. All model parameters are randomly initialised and trained from scratch without pretrained language models.

2.2 Architecture Selection

We compare three sequence-modelling architectures: attention-based BiGRU, attention-based BiLSTM, and a Transformer encoder. The models use the same task formulation, data split, and evaluation procedure, with other experimental settings kept broadly consistent.

All three models achieved similar preliminary accuracies of approximately 60%. However, the Transformer exhibited a slower increase in the observed overfitting gap during training. We therefore selected it as the primary architecture for subsequent experiments. This decision was based on observed training behaviour rather than a substantial initial accuracy advantage.

Figure 1 illustrates the two encoder architectures: recurrent encoders (GRU/LSTM) and the Transformer encoder. Despite their architectural differences, all models use the same candidate-scoring framework, in which each candidate is encoded independently and scored by a shared linear layer.

[Insert Figure 1 here: (Left) RNN, (Right) Transformer.]

2.3 Transformer Architecture

The Transformer classifier consists of a trainable token embedding layer, optional sinusoidal positional encoding, stacked Transformer encoder layers, a pooling operation, and a linear scoring head. Its default configuration uses an embedding dimension of 64, four attention heads, two encoder layers, a feed-forward dimension of 128, and a dropout rate of 0.1.

Given an input sequence of length $L$, the embedding layer produces $\mathbf{E} \in \mathbb{R}^{L \times d}$, where $d$ is the embedding dimension. When positional encoding is enabled, fixed sinusoidal encodings are added to the token embeddings:

$$
\begin{aligned}
PE_{(p,2i)} &= \sin\left(\frac{p}{10000^{2i/d}}\right),\\
PE_{(p,2i+1)} &= \cos\left(\frac{p}{10000^{2i/d}}\right).
\end{aligned}
$$

Here, $p$ denotes the token position and $i$ indexes the embedding dimensions. These encodings are stored as non-trainable buffers. Dropout is applied before the representations enter the encoder.

Each Transformer encoder layer applies multi-head self-attention followed by a position-wise feed-forward network, with residual connections, layer normalisation, and dropout. Scaled dot-product attention is defined as

$$
\operatorname{softmax}\left(
\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}+\mathbf{M}
\right)\mathbf{V},
$$

where $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ are the query, key, and value matrices, $d_k$ is the key dimension, and $\mathbf{M}$ masks padded key positions. Multi-head attention applies several learned projections in parallel and combines their outputs to capture different contextual relationships.

Unlike recurrent models, the Transformer does not rely on sequential recurrence to propagate information between tokens. Instead, self-attention allows each valid token to incorporate information from other valid tokens. After encoding, the sequence is reduced to a fixed-dimensional vector using either the first token's representation ($[CLS]$) or masked mean pooling. The default configuration uses $[CLS]$ pooling. The pooled representation is passed through dropout and a linear layer to produce a scalar candidate score. Padding masks prevent padded positions from being treated as valid attention keys.

2.4 Recurrent Baselines and Shared Candidate Scoring

The BiGRU and BiLSTM baselines use trainable token embeddings followed by bidirectional recurrent encoders. For each token, the forward and backward hidden states are concatenated:

$$
\mathbf{h}_t =
[\overrightarrow{\mathbf{h}}_t;
\overleftarrow{\mathbf{h}}_t].
$$

This representation incorporates information from both preceding and following tokens. The default recurrent configuration uses an embedding dimension of 64 and a hidden dimension of 128 per direction, resulting in a 256-dimensional output. Sequences are packed according to their valid lengths to avoid unnecessary recurrent computation on padding tokens.

Attention mechanisms have been used to construct informative sequence representations in recurrent models. For example, Yang et al. [2] demonstrated the effectiveness of attention-based recurrent architectures across multiple text classification tasks. In our implementation, a single-head scaled dot-product self-attention module operates on the contextual representations produced by the bidirectional recurrent encoder.

Given the recurrent representation matrix $\mathbf{H}$, the module computes

$$
\mathbf{Q} = \mathbf{H}\mathbf{W}_Q,\quad
\mathbf{K} = \mathbf{H}\mathbf{W}_K,\quad
\mathbf{V} = \mathbf{H}\mathbf{W}_V,
$$

followed by

$$
\mathbf{A} =
\operatorname{softmax}\left(
\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_a}}+\mathbf{M}
\right),\qquad
\widetilde{\mathbf{H}} = \mathbf{A}\mathbf{V},
$$

where $d_a$ is the recurrent output dimension and $\mathbf{M}$ masks padded key positions. The attention-processed states are aggregated using masked mean pooling:

$$
\mathbf{r} =
\frac{\sum_{t=1}^{L}m_t\widetilde{\mathbf{h}}t}
{\max\left(1,\sum{t=1}^{L}m_t\right)},
$$

where $m_t \in {0,1}$ indicates whether token $t$ is valid. When attention is disabled, the model instead concatenates the final forward and backward hidden states. Consequently, this ablation compares the overall representation and pooling designs rather than isolating the effect of attention alone.

Although both recurrent and Transformer models employ scaled dot-product attention, their roles differ. Recurrent attention operates on representations that already incorporate sequential context, whereas Transformer self-attention contextualises token representations throughout the encoder.

All three architectures use a shared candidate-scoring framework. Let $\mathbf{r}_1$ and $\mathbf{r}_2$ denote the representations of the two candidates. A shared linear layer computes

$$
z_j = \mathbf{w}^{\top}\mathbf{r}_j + b,
\qquad j \in {1,2}.
$$

The scores are concatenated into $\mathbf{z} = [z_1,z_2]$. The predicted class is the index of the larger score, with class 0 corresponding to the first candidate and class 1 to the second. The models are trained using cross-entropy loss against the gold label. Dropout is applied to embeddings and/or pooled representations for regularisation.

2.5 Design Rationale and Limitations

The shared encoder and scoring head provide a consistent scoring procedure for both candidates without requiring separate parameters for the first and second solution positions. Goal-conditioned input construction enables each solution to be evaluated in the context of the corresponding goal. The recurrent baselines model sequential dependencies through bidirectional recurrence, while the Transformer provides direct token-to-token interactions through multi-head self-attention.

The Transformer architecture and scaled dot-product attention follow the design introduced by Vaswani et al. [3]. Our implementation adapts this architecture to PIQA through goal–solution sequence construction, shared candidate scoring, and padding-aware processing. The recurrent baselines use a related scaled dot-product attention formulation, enabling a comparison between attention applied after recurrent encoding and attention integrated throughout the encoder.

Several limitations remain. First, the two candidates are encoded independently, without explicit cross-candidate interaction. Second, the preliminary results showed similar accuracies across the three architectures, so the Transformer did not demonstrate a clear initial accuracy advantage. Its selection was instead motivated partly by its observed training behaviour. Finally, attention weights describe how representations are combined but do not necessarily provide faithful explanations of model decisions or establish genuine physical understanding.

# Experimental Setup

## Dataset description

3.1 Dataset and Preprocessing

We use the PIQA dataset, which contains a goal and two candidate solutions for each example. The dataset presents several challenges for sequence modelling: Input lengths vary, candidate solutions may differ in subtle lexical details, and some goals occur in multiple examples with different candidate pairs.

We adopt a conservative text-normalisation strategy to minimise unnecessary changes to the original text. 
Text preprocessing consists of Unicode NFKC normalisation, replacement of consecutive whitespace characters with a single ASCII space, and removal of leading and trailing whitespace. We retain the original casing, punctuation, and stop words because short functional words and minor lexical differences may affect the interpretation of a candidate solution. We do not apply stemming, lemmatisation, or stop-word removal.

The cleaned text is tokenised using SentencePiece. This subword tokenisation approach allows the models to process text as sequences of token IDs while retaining punctuation and case information. Special tokens are used to construct the model inputs, and padding is applied during batch collation. Each candidate is represented as ($[CLS]\ g\ [SEP]\ s_j\ [EOS]$), where ($g$) is the goal and ($s_j$) is a candidate solution.


3.2 Data Splitting and Leakage Prevention

We divide the supplied training data into training and validation subsets using a goal-based split. Unique goals are shuffled using a fixed random seed of 4012. Approximately 10% of the unique goals are assigned to validation, and all examples associated with those goals are placed in the validation subset.

This strategy prevents identical goals from appearing in both the training and validation subsets. However, it does not by itself establish that goals are disjoint between the training data and the held-out test set. The validation subset is used to monitor training and compare configurations, while the test set is reserved for final evaluation.

3.3 Batching and Padding Efficiency

The two candidate sequences are padded independently within each batch. Boolean masks identify valid tokens and padding positions, allowing models to exclude padding from attention computations and masked pooling where applicable.

To reduce padding overhead, we implement a length-based bucket batch sampler. It estimates each example's maximum candidate length using tokenised sequence lengths and the relevant special-token overhead. After shuffling examples, it groups them into larger windows, sorts examples by estimated length within each window, and constructs mini-batches from neighbouring examples. The resulting batch order is then shuffled.

In our measurements, length-based bucketing reduced the padding ratio from 86.39% to 34.17%, a reduction of 52.22 percentage points. This indicates substantially less padding under the evaluated batching configuration, although the actual speed improvement must be measured separately.

3.4 GPU Data Preloading

The prepared training data are relatively small, occupying approximately 4.2 MB before batching. We therefore preload the constructed training batches onto the GPU when memory capacity permits. This reduces repeated CPU-to-GPU data transfers during training and allows the training loop to access its batches directly from device memory.

The same approach can be applied to validation and test batches where appropriate. Validation data are used to monitor training, while test metrics are computed for final evaluation. GPU preloading is an implementation optimisation and does not change the model architecture or learning objective.



# References so far

[1] Y. Bisk, R. Zellers, R. Le Bras, J. Gao, and Y. Choi, “PIQA: Reasoning about physical commonsense in natural language,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 5, pp. 7432–7439, 2020, doi: 10.1609/aaai.v34i05.6239.

[2] Z. Yang, D. Yang, C. Dyer, X. He, A. Smola, and E. Hovy, “Hierarchical attention networks for document classification,” in Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2016, pp. 1480–1489, doi: 10.18653/v1/N16-1174.

[3] A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017.


去 google scholar 搜索可以获得文献信息

https://ojs.aaai.org/index.php/AAAI/article/view/6239

```bibtex
@article{bisk2020piqa,
  author  = {Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, Yejin Choi},
  title   = {{PIQA}: Reasoning about Physical Commonsense in Natural Language},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {34},
  number  = {5},
  pages   = {7432--7439},
  year    = {2020},
  doi     = {10.1609/aaai.v34i05.6239}
}
```

https://aclanthology.org/N16-1174.pdf

```bibtex
@inproceedings{yang2016hierarchical,
  author    = {Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy},
  title     = {Hierarchical Attention Networks for Document Classification},
  booktitle = {Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages     = {1480--1489},
  year      = {2016},
  doi       = {10.18653/v1/N16-1174}
}
```

https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

```bibtex
@inproceedings{NIPS2017_3f5ee243,
 author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, \L ukasz and Polosukhin, Illia},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. Von Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Attention is All you Need},
 url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf},
 volume = {30},
 year = {2017}
}
```







--------------------------------------------------------------------------------------------

## Training procedure and hyperparameters




## Baseline models




## Evaluation metrics




# Results and Analysis


# Results and Analysis


## Ablation studies


## Qualitative analysis of attention


# Conclusion


# Team Contributions

(This section does not count towards the page limit.)

# References

(This section does not count towards the page limit.)

Generate your references using BibTeX.

