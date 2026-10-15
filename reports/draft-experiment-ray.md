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