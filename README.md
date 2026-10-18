# PIQA Commonsense Reasoning

For details and a runnable demonstration, see:

- Report: [`Report`](./reports/CITS4012_67.pdf).

- Runnable demonstration: [`Notebook`](CITS4012_67.ipynb).

Physical commonsense reasoning requires models to reason about everyday physical interactions beyond surface-level linguistic patterns. The PIQA benchmark evaluates this ability by asking models to select the more physically plausible of two solutions for a given goal.

This project compares BiGRU, BiLSTM, and Transformer encoders trained from scratch, alongside rule-based and pretrained-model baselines. The Transformer achieves 58.9% test accuracy, outperforming the length heuristic but remaining around 11 percentage points below leading pretrained models.

Attention analysis suggests that the model can identify differences between candidate solutions but often lacks the physical knowledge needed to judge their plausibility. These findings highlight the limitations of learning physical commonsense from language alone.



## Methodology

### Data Pipeline

- Dataset: PIQA, with 14,507 training, 1,606 validation, and 1,838 test examples.

- Preprocessing: Unicode NFKC normalisation and whitespace standardisation.

- Tokenisation: SentencePiece BPE with an 8,000-token vocabulary.

- Input format: `[CLS] goal [SEP] solution [EOS]`.

- Efficiency: Goal-level training/validation splitting, length-bucketed batching, Boolean padding masks, and optional GPU data preloading.

### Main Model

A Transformer encoder trained from scratch to score two candidate solutions using shared parameters and cross-entropy loss. The final configuration uses AdamW with a learning rate of (9\times10^{-4}), an embedding dimension of 100, two encoder layers, four attention heads, dropout of 0.3, and a batch size of 512.

Models are trained for 300 epochs, retaining the checkpoint with the highest validation accuracy. Experiments use three random seeds: 4012, 4013, and 4014.

### Baselines

| ID | Method | Purpose |
|---|---|---|
| B0 | Shorter-solution heuristic | Tests a simple character-length rule |
| B1 | BiLSTM without attention | Compares recurrent encoding without attention |
| B2 | Zero-shot GPT-2 and Qwen2.5-0.5B | Evaluates pretrained language-model scoring |
| B3 | Fine-tuned DistilBERT and RoBERTa-base | Evaluates pretrained encoder representations |

### Ablation Experiments

Each trained ablation uses the final configuration without hyperparameter retuning.

| ID | Modification | Purpose |
|---|---|---|
| A1′-a | Remove positional encoding | Tests the contribution of positional information |
| A1′-b | Reduce attention heads from four to one | Tests the effect of multi-head attention |
| A3 | Set dropout to zero | Tests the effect of dropout regularisation |
| A2-a | Remove the goal at evaluation time | Tests the model's dependence on goal context |

The three trained ablations use seeds `4012`–`4014` and 300 epochs. A2-a uses the existing final-model checkpoints and does not require retraining. Other proposed ablations were not conducted.

## Experimental Workflow

1. Preprocess and tokenise the PIQA dataset.

2. Evaluate heuristic, recurrent, and pretrained-model baselines.

3. Select the main architecture and tune hyperparameters using validation data.

4. Train the final model with three random seeds and retain the best validation checkpoint for each run.

5. Conduct ablation experiments without retuning.

6. Evaluate accuracy, compare paired predictions using continuity-corrected McNemar tests, and inspect attention patterns and prediction errors.

The test set is reserved for final evaluation.

## Environment and Reproducibility

Experiments were conducted in Google Colab with GPU acceleration, using NVIDIA Tesla T4 or A100 GPUs depending on availability.

To reproduce the experiments, open ['Notebook'](CITS4012_67.ipynb) in Google Colab, select a GPU runtime, and run the notebook cells in order.

For manual reproduction, ensure that the required dependencies, PIQA dataset, and SentencePiece model are available. Use the specified random seeds and configurations, and verify that saved checkpoints and results correspond to the current code revision.

## Further Reading

See the [`Report`](./reports/CITS4012_67.pdf) for the full experimental setup, hyperparameter selection, quantitative comparisons, ablation results, attention visualisations, and limitations.


