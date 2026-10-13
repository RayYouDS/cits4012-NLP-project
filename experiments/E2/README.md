# E2 — Hyperparameter tuning of the main model (M4)

This folder holds the E2 notebook, the frozen configuration used for the main model, and the
per-run result summaries. Only validation data was used. The test set was not evaluated in E2.

## Final decision

| Item | Value |
|---|---|
| Optimiser | AdamW, weight_decay 0.01 |
| Learning rate | 0.0009 |
| Dropout | 0.3 |
| Layers | 2 (heads 4, embedding dim 100, pooling cls, positional encoding on) |
| Batch / epochs / early stopping | 512 / 300 / none |
| Validation accuracy (best epoch), 3 seeds (4012, 4013, 4014) | **0.5940 ± 0.8 pp** (sample s.d.) |
| Plateau mean validation accuracy, 3 seeds | 0.5797 (per seed 0.5874, 0.5805, 0.5711) |

Frozen configuration: `config/e2_final_300.json` (copied from `results/final_3seed/config.json`,
which records `_commit` d0268525, i.e. commit d026852 on `baselines`).

The best-epoch figure (0.5940) is an optimistic estimate, because it takes the best of 300 noisy epochs.
The plateau mean (0.5797) is the more conservative estimate of the same model. Use both when comparing E2 with E1 or with the baselines.

## Protocol

- Data: goal-grouped split (train 14,507 / valid 1,606 / test 1,838), seed 4012, tokenizer `piqa_bpe_v3.model`.
- Model: `TransformerClassifier`; `run_m3.py` training loop with best-epoch restoration.
- Budget: 300 epochs, no early stopping, batch 512. The budget matches the E1 run recorded as
  `e1_stress_results_new_transformer`. Momentum 0.9 is set in the config for all runs; it only affects SGD.
- Staged search, one variable per stage, each stage starts from the previous stage's winner. Stages 1–3 use seed 4012 only.
- Selection: the plan (section 2) does not define a selection rule for E2. We applied the E1 decision rule from plan section 1
  by analogy: highest validation accuracy, unless McNemar on the two top candidates is not significant (alpha 0.05).
  Then the E1 tie rule applies: smaller overfitting gap, then shorter time to best, then fewer parameters.
  Every override is recorded below.

## Metrics used in the tables

- **Valid acc**: validation accuracy at the best epoch (the best epoch's weights are restored before evaluation).
- **Best epoch**: the epoch with the highest validation accuracy.
- **Gap at best**: train-eval accuracy minus validation accuracy, both at the best epoch. Larger means more overfitting.
- **Plateau mean**: mean validation accuracy over the second half of training. A check that the peak is not a single noisy epoch.

## Stage 1 — optimiser and learning rate (1 seed, dropout 0.1, 3 layers)

Grid: Adam and AdamW at E1's best learning rate (3e-4) divided by 3, times 1, times 3; SGD at 0.01, 0.03, 0.1.

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

Top two: AdamW 9e-4 vs Adam 9e-4, difference 0.69 pp, McNemar χ² = 0.297, p = 0.586 (not significant).
E1 tie rule: smaller gap (0.389 vs 0.398) and shorter time to best (153 vs 233) both favour AdamW 9e-4.
**Decision: AdamW 9e-4 (no override needed).**

Notes:
- The best learning rate lies on the upper edge of the grid (3× E1's value). Larger rates were not tested.
- AdamW 3e-4 (best epoch 296), Adam 3e-4 (295) and SGD 0.01 (293) peaked at the 300-epoch cap, so these runs
  may have been budget-limited. The comparison is fair for the runs that peaked well before the cap, not for these.

## Stage 2 — dropout (AdamW 9e-4, 3 layers, 1 seed)

| Dropout | Valid acc | Best epoch | Gap at best | Plateau mean |
|---|---|---|---|---|
| 0.0 | 0.5791 | 127 | 0.420 | 0.5559 |
| 0.1 | 0.6090 | 153 | 0.389 | 0.5862 |
| 0.3 | 0.5971 | 204 | **0.298** | 0.5793 |

Top two: dropout 0.1 vs 0.3, difference 1.19 pp, McNemar χ² = 0.694, p = 0.405 (not significant).
E1 tie rule: dropout 0.3 has the smaller overfitting gap (0.298 vs 0.389).
**Decision: dropout 0.3. Recorded as `PICK[3] = 's2_dropout0.3'`** (applied when stage 3 was built).

## Stage 3 — depth (AdamW 9e-4, dropout 0.3, 1 seed)

| Layers | Valid acc | Best epoch | Gap at best | Plateau mean | Parameters |
|---|---|---|---|---|---|
| 2 | **0.6040** | 203 | **0.295** | 0.5877 | 933,357 |
| 3 | 0.5971 | 204 | 0.298 | 0.5793 | 999,985 |

2 layers has the smaller overfitting gap (0.295 vs 0.298), so the E1 tie rule selects it directly; it also has
fewer parameters and higher validation accuracy. The McNemar test for this 2-layer vs 3-layer pair was **not run**.
Note that the gap difference (0.003) is far below the noise level, so this decision is weakly supported.
**Decision: 2 layers.**

## Final — three seeds for the chosen configuration

| Seed | Valid acc | Best epoch | Gap at best |
|---|---|---|---|
| 4012 | 0.5996 | 213 | 0.290 |
| 4013 | 0.5978 | 179 | 0.268 |
| 4014 | 0.5847 | 164 | 0.267 |
| **Mean ± s.d.** | **0.5940 ± 0.0081** | | mean gap 0.275 |

**Override:** ranking the stage 1–3 runs by validation accuracy alone would select the stage 1 configuration
(AdamW 9e-4, dropout 0.1, 3 layers; 0.6090). That configuration was not the one carried forward: stage 2 chose dropout 0.3
by the E1 tie rule, so the stage 1 winner is not on the staged path. `PICK['final'] = 's3_layers2'` was set to keep the staged decisions consistent.

Test of the override pair (run in the notebook's results cell, not in the stage 3 decision):
s1_adamw_lr0.0009 (0.6090) vs s3_layers2 (0.6040), on the same 1606 validation questions:
only A correct = 251, only B correct = 243, χ² = 0.099, **p = 0.753 (not significant)**.
The two are statistically indistinguishable. The choice of s3_layers2 rests on the E1 tie rule
(gap 0.295 vs 0.389), not on accuracy.

Note on the single-seed numbers: the stage 1–3 winners were selected from single runs, which
overstates them. The three-seed mean (0.5940) is the best-epoch estimate for the frozen model; the plateau mean (0.5797) is the more conservative one.

## Caveats

1. **Single seed per stage.** Differences below about 1.25 pp (one standard error) are not reliable. Stage
   decisions are therefore based on the E1 decision rules, not on accuracy alone.
2. **Run-to-run variation on the same configuration and seed.** Seed 4012 gave 0.6040 in stage 3 (2 layers)
   and 0.5996 in the final run with an identical configuration, a difference of 0.44 pp. GPU nondeterminism
   is the likely cause. Differences of this size between stages are noise.
3. **Learning-rate grid edge.** The best rate (9e-4) is at the top of the planned grid.
4. **Training budget.** The plan called for a 1000-epoch stress test to find a plateau. E2 did not run it; it uses the
   300-epoch budget of the E1 run. Confirm with the team that E1 set this budget. Three stage 1 runs peaked at epochs 293–296 of 300,
   so their results may be budget-limited (see Stage 1 notes).
5. **Plateau and best epoch disagree on the stage winner.** The plateau mean ranks `s3_layers2` (0.5877) above
   `s1_adamw_lr0.0009` (0.5862), while the best-epoch rule ranks them the other way (0.6090 vs 0.6040). The notebook
   flags this disagreement. Plateau means of the leading candidates differ by less than 0.2 pp, so this stage does not separate them clearly.
6. **Overfitting is large in every run.** Gaps are 0.27–0.42 for the stronger runs. The final model's train-eval accuracy is about 0.85–0.89, against about 0.59 on validation.
7. **Parameter count.** The frozen model has 933,357 parameters.

## Deviations from the plan

- Plan section 2 estimated about 12 runs (8 in stage 1, 2 in stage 2, 1–2 in stage 3, 2 extra seeds). The actual total is 15.
- Stage 1 ran all 9 learning-rate runs. The plan allowed reusing E1's Adam 3e-4 result. That result was
  reproduced here at 0.5922 (E1: 0.5903), so the rerun is consistent with it.
- The final stage re-ran seed 4012 (the plan asked for two additional seeds only). Cost: about 16 minutes.
- Total runs: 9 (stage 1) + 2 (stage 2) + 1 (stage 3) + 3 (final) = 15.
- The plan's 1000-epoch stress test was not part of E2 (see Caveat 4).
- Model weights (`best.pt`) are on Google Drive, not in the repository (see Files).

## Files

- `E2_tuning_300.ipynb`: notebook that produced all runs. Clones `baselines` and records `_commit` in each `config.json`.
  Saved outputs of the results cell are included.
- `config/e2_final_300.json`: frozen configuration for E4 and later experiments.
- `results/<run>/`: for each run, `config.json`, `m3_summary.csv`, `m3_transformer_*_history.csv`
  and `m3_transformer_*_val_predictions.csv`. Model weights (`*.pt`) are kept on Google Drive
  (`MyDrive/cits4012/e2_full_300/`), not in the repository. The plan asks for `best.pt` per run; it exists on Drive only.

## Reproduction

Run the notebook with `STAGE = 1`, then `2`, `3`, and `'final'`, setting the `PICK` entries shown above
before stages 3 and final. Stage 1 must finish before stage 2 is set.
