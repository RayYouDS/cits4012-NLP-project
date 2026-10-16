# E2 — Hyperparameter tuning of the main model (M4)

This folder holds the E2 notebook, the frozen configuration of the main model, the saved results of every
run, and the trained weights of the final model. Only validation data was used. The test set was not evaluated in E2.

## Contents

| Path | What it is |
|---|---|
| `E2_tuning_300.ipynb` | The notebook. Read mode (default) shows the stage decisions, the results table, the training curves and how to load the model. |
| `config/e2_final_300.json` | A copy of the final run's configuration (`_commit` d0268525). The notebook writes the frozen values itself and checks them against `results/final_3seed/config.json`. |
| `results/e2_stage_table.csv` | All 15 runs in one table: stage, settings, validation accuracy, plateau mean, gap, best epoch, parameters. |
| `results/e2_final_curves.png` | Validation accuracy and overfitting gap per epoch for the three final seeds, drawn from the saved histories. |
| `results/<run>/` | One folder per run: `config.json`, `m3_summary.csv`, the per-epoch `history.csv`, and for some runs the validation predictions and plots. |
| `checkpoints/` | The trained final model for seeds 4012, 4013 and 4014 (`*_best.pt`, PyTorch `state_dict`, 4.6 MB each). |

## Final decision

| Item | Value |
|---|---|
| Optimiser | AdamW, weight_decay 0.01 |
| Learning rate | 0.0009 |
| Dropout | 0.3 |
| Layers | 2 (heads 4, embedding dim 100, pooling cls, positional encoding on) |
| Batch / epochs / early stopping | 512 / 300 / none |
| Parameters | 933,357 |
| Validation accuracy (best epoch), 3 seeds (4012, 4013, 4014) | **0.5940 ± 0.0081** (sample s.d.) |
| Plateau mean validation accuracy, 3 seeds | 0.5797 (per seed 0.5874, 0.5805, 0.5711) |

The best-epoch figure is an optimistic estimate, because it takes the best of 300 noisy epochs. The plateau mean is the more
conservative estimate of the same model. Use both when comparing with E1 or with the baselines.

## Protocol

- **Data**: goal-grouped split, seed 4012, tokenizer `piqa_bpe_v3.model`. Training 14,507 and validation 1,606 questions, as in the saved log.
  The test split (1,838 questions) is stated in the earlier version of this README; it is not loaded anywhere in E2.
- **Model**: `TransformerClassifier`, trained with the training loop in `scripts/training/run_m3.py`, which restores the best epoch's weights before evaluation.
- **Budget**: 300 epochs, no early stopping, batch 512, for every run. The budget is the one used by the E1 run (`e1_stress_results_new_transformer`).
  Momentum 0.9 is recorded in every config; it only affects SGD.
- **Staged search**: one variable per stage, each stage starting from the previous stage's winner. Stages 1–3 use seed 4012 only.
- **Selection**: the plan (section 2) does not define a rule for E2. We applied the E1 rule from plan section 1 by analogy: highest
  validation accuracy, unless McNemar on the two top candidates is not significant (alpha 0.05). Then the E1 tie rule applies:
  smaller overfitting gap, then shorter time to best, then fewer parameters. Every override is recorded below.

## Metrics

- **Valid acc**: validation accuracy at the best epoch (the best epoch's weights are restored before evaluation).
- **Best epoch**: the epoch with the highest validation accuracy.
- **Gap at best**: train-eval accuracy minus validation accuracy, both at the best epoch. Larger means more overfitting.
- **Plateau mean**: mean validation accuracy over the second half of training. A check that the peak is not a single noisy epoch.

## Stage 1 — optimiser and learning rate (1 seed, dropout 0.1, 3 layers)

Grid: Adam and AdamW at 3e-4 divided by 3, times 1 and times 3; SGD at 0.01, 0.03 and 0.1.

| Run | Valid acc | Best epoch | Gap at best | Plateau mean |
|---|---|---|---|---|
| **AdamW 9e-4** | **0.6090** | 153 | 0.389 | 0.5862 |
| Adam 9e-4 | 0.6021 | 233 | 0.398 | 0.5876 |
| AdamW 3e-4 | 0.5996 | 296 | 0.394 | 0.5875 |
| SGD 0.1 | 0.5971 | 191 | 0.359 | 0.5718 |
| Adam 3e-4 | 0.5922 | 295 | 0.400 | 0.5773 |
| Adam 1e-4 | 0.5897 | 116 | 0.195 | 0.5741 |
| AdamW 1e-4 | 0.5872 | 123 | 0.199 | 0.5740 |
| SGD 0.03 | 0.5797 | 257 | 0.267 | 0.5571 |
| SGD 0.01 | 0.5592 | 293 | 0.042 | 0.5329 |

Top two: AdamW 9e-4 vs Adam 9e-4, difference 0.69 pp, McNemar χ² = 0.297, p = 0.586 (not significant; recomputed from the saved predictions).
E1 tie rule: the smaller gap (0.389 vs 0.398) and the shorter time to best (153 vs 233) both favour AdamW 9e-4.
**Decision: AdamW 9e-4.**

Notes:
- The best learning rate is at the upper edge of the grid. Larger rates were not tested.
- AdamW 3e-4 (best epoch 296), Adam 3e-4 (295) and SGD 0.01 (293) peaked near the 300-epoch cap, so they may have been budget-limited.

## Stage 2 — dropout (AdamW 9e-4, 3 layers, 1 seed)

| Dropout | Valid acc | Best epoch | Gap at best | Plateau mean |
|---|---|---|---|---|
| 0.0 | 0.5791 | 127 | 0.420 | 0.5559 |
| 0.1 | 0.6090 | 153 | 0.389 | 0.5862 |
| 0.3 | 0.5971 | 204 | **0.298** | 0.5793 |

Top two: dropout 0.1 vs 0.3, difference 1.19 pp, McNemar χ² = 0.694, p = 0.405 (not significant; recomputed).
E1 tie rule: dropout 0.3 has the smaller overfitting gap (0.298 vs 0.389).
**Decision: dropout 0.3**, recorded as `PICK[3] = 's2_dropout0.3'`.

## Stage 3 — depth (AdamW 9e-4, dropout 0.3, 1 seed)

| Layers | Valid acc | Best epoch | Gap at best | Plateau mean | Parameters |
|---|---|---|---|---|---|
| 2 | **0.6040** | 203 | **0.295** | 0.5877 | 933,357 |
| 3 | 0.5971 | 204 | 0.298 | 0.5793 | 999,985 |

The 3-layer row is the stage 2 dropout 0.3 run, which has the same configuration.
2 layers has the smaller overfitting gap (0.295 vs 0.298), so the E1 tie rule selects it; it also has fewer parameters and higher validation accuracy.
The McNemar test for this pair was **not run**. The gap difference (0.003) is far below the noise level, so this decision is weakly supported.
**Decision: 2 layers.**

## Final — three seeds for the chosen configuration

| Seed | Valid acc | Best epoch | Gap at best | Time (s) |
|---|---|---|---|---|
| 4012 | 0.5996 | 213 | 0.290 | 644 |
| 4013 | 0.5978 | 179 | 0.268 | 647 |
| 4014 | 0.5847 | 164 | 0.267 | 648 |
| **Mean ± s.d.** | **0.5940 ± 0.0081** | | mean gap 0.275 | |

The notebook log records 32.5 minutes for the three seeds.

**Override.** Ranking the stage 1–3 runs by validation accuracy alone would select the stage 1 configuration (AdamW 9e-4, dropout 0.1, 3 layers; 0.6090). That configuration is not on the staged path, because stage 2 chose dropout 0.3 by the tie rule. `PICK['final'] = 's3_layers2'` keeps the decisions consistent.

Test of the override pair (the notebook's results cell, not part of the stage decision): s1_adamw_lr0.0009 (0.6090) vs s3_layers2 (0.6040) on the same 1,606 questions:
only A correct = 251, only B correct = 243, χ² = 0.099, **p = 0.753 (not significant)**. The two are statistically indistinguishable. The choice of s3_layers2 rests on the tie rule (gap 0.295 vs 0.389), not on accuracy.

## Caveats

1. **One seed per stage.** Differences below about 1.25 pp (one standard error on 1,606 questions) are within noise. Stage decisions rest on the tie rules, not on accuracy alone.
2. **Run-to-run variation.** Seed 4012 gave 0.6040 in stage 3 (2 layers) and 0.5996 in the final run, with the same configuration: a difference of 0.44 pp. The cause is likely GPU nondeterminism; this was not confirmed.
3. **Training loader is preloaded.** Each run's training batches are fixed in one list before training, so every epoch sees the same batches in the same order. Seeds still change the order between runs.
4. **Budget.** The plan called for a 1000-epoch stress test to find a plateau. E2 did not run it; it uses the 300-epoch budget of the E1 run. Confirm with the team that E1 set this budget.
5. **Plateau and best epoch disagree on the stage 3 winner.** The plateau mean ranks `s3_layers2` (0.5877) above `s1_adamw_lr0.0009` (0.5862); the best-epoch rule ranks them the other way (0.6090 vs 0.6040). The two plateau means differ by 0.15 pp.
6. **Overfitting is large in every run.** Training accuracy in evaluation mode is about 0.85–0.89 for the final model, against about 0.59 on validation.
7. **Learning-rate grid edge.** The best rate (9e-4) is at the top of the planned grid.

## Deviations from the plan

- Stage 1 ran all nine learning-rate runs. The Adam 3e-4 run reproduced at 0.5922; E1's own result for that setting is not in this folder, so the two are not compared here.
- The final stage re-ran seed 4012 as well as seeds 4013 and 4014. Each seed took about 11 minutes (643.8 s in the summary).
- The plan's 1000-epoch stress test was not part of E2 (see caveat 4).
- Total runs: 9 (stage 1) + 2 (stage 2) + 1 (stage 3) + 3 (final) = 15.

## Reproduction

**Read mode (default, for marking):** open `E2_tuning_300.ipynb` and run all cells from a downloaded or cloned copy of the repository. It needs no GPU, Google Drive or internet. The cell that loads the saved model needs the repository code and the PIQA data, both in the repository. It reproduces the saved validation accuracy for seed 4012 (0.599626, difference 0.000000).

**Training mode:** set `RUN_TRAINING = True` and `STAGE` to 1, 2, 3 and then `'final'`, in Colab with a T4 GPU, setting `PICK` as shown above before stages 3 and final. Stage 1 must finish before stage 2 starts.

## Environment

Training (Colab, the runs in `results/`): Tesla T4, PyTorch 2.11.0+cu130, code commit `d0268525` on `baselines`.
