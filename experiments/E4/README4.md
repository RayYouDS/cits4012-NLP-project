# E4: Ablation experiments on the main model (M6)

This folder holds the E4 notebook (`E4_ablation1 (2).ipynb`, executed on Colab), the result tables, the figure,
the environment record and the run logs. All numbers are on the **validation set**. The test set is not loaded or evaluated here.

## Reference model

The main model is the frozen E2 configuration (`config/e2_final_300.json`, from E2 `_commit` d0268525):
Transformer, 2 layers, 4 heads, embedding dim 100, dropout 0.3, AdamW lr 9e-4, weight decay 0.01,
batch 512, 300 epochs, no early stopping, 933,357 parameters.
Its validation accuracy over seeds 4012, 4013, 4014 is 0.5940 ± 0.0081 (best epoch).

## What was run

| ID | Change from the main model | Question | Status |
|---|---|---|---|
| A1'-a | No positional encoding (`a1_no_position`) | Does position information help accuracy? | Trained, 3 seeds |
| A1'-b | One attention head instead of four (`a1_one_head`) | Do multiple heads help? | Trained, 3 seeds |
| A3 | Dropout 0 instead of 0.3 (`a3_no_dropout`) | Does dropout reduce overfitting? | Trained, 3 seeds |
| A2-a | Same trained main checkpoints, goal removed at **evaluation** only | Does the trained model depend on the goal? | Evaluated, 3 seeds, no retraining |

Not run:
- A1 (RNN with no self-attention): not applicable, because the main model is a Transformer.
- A1'-c (CLS vs mean pooling): optional, not run.
- A2-b (trained without goal): optional, not run.
- B1 (no-attention baseline): owned by a teammate, not part of E4.

## Results (validation accuracy, mean ± sample s.d. over 3 seeds)

| Experiment | Valid acc | Δ vs main (pp) | Per-seed McNemar p (4012 / 4013 / 4014) | Reading |
|---|---|---|---|---|
| Main (E2) | 0.5940 ± 0.0081 | — | — | — |
| A1'-a no position | 0.5969 ± 0.0050 | +0.29 | 0.88 / 0.96 / 0.57 | No significant difference |
| A1'-b one head | 0.5863 ± 0.0016 | −0.77 | 0.24 / 0.33 / 0.83 | No significant difference; direction not consistent |
| A3 no dropout | 0.5627 ± 0.0121 | −3.13 | 0.14 / 0.006 / 0.064 | Lower in all 3 seeds |
| A2-a no goal (eval only) | 0.5702 ± 0.0051 | −2.39 | 0.033 / 0.003 / 0.099 | Lower in all 3 seeds |

Δ is ablation minus main. McNemar tests are paired per seed on the 1,606 validation questions.

A3 overfitting (primary metric, from the plan):

| | Gap at best epoch | Gap at epoch 300 | Valid loss at epoch 300 | Best epoch |
|---|---|---|---|---|
| Main | 0.275 ± 0.013 | 0.347 ± 0.005 | 1.05 | 185 |
| A3 no dropout | 0.255 ± 0.187 | 0.459 ± 0.021 | 6.88 | 11 |

At the best epoch the gap does not separate the two models, because the no-dropout model peaks at about
epoch 11, before it has overfitted. The gap at epoch 300 and the validation loss show the overfitting.
The no-dropout run also uses an effectively early-stopped checkpoint, so its accuracy gap mixes the
effect of dropout with the effect of training length.

A2-a: with goal removed, 302, 268 and 249 of 1,606 predictions change (seeds 4012, 4013, 4014).
The changes go both ways, but more answers are lost without the goal than gained.
The model was never trained without the goal, so this measures sensitivity to input change, not
reasoning without the goal.

## Environment

Recorded by the last cell of `E4_ablation1 (2).ipynb` (`env.json`, `env_freeze.txt`):

| Item | Value |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| GPU | Tesla T4 |
| E4 code commit (`baselines`) | 2fa9820ccd9e0c8833ffbf6ec900c5d7f0f7cd94 |
| E2 code commit (main model) | d0268525b83d7af9f5d5b3a684b6ad8dec00d4d7 |

## Files

Top level of this folder (`experiments/E4/`):

- `E4_ablation1 (2).ipynb`: the executed notebook. Trains or verifies each run, runs A2-a, produces the tables, the figure and the environment record.
- `README4.md`: this file.
- `e4_table1.csv`: per-experiment means and s.d. (valid accuracy, gaps, validation loss, best epoch).
- `e4_table2.csv`: paired comparison with each ablation (Δ, per-seed McNemar p).
- `e4_a2a_summary.csv`: A2-a mean and s.d.
- `a3_gap_loss.png`: validation loss and overfitting gap curves, main vs A3, per seed.
- `env.json`, `env_freeze.txt`: runtime and package versions (see Environment).

Run logs (`e4_logs/`), copied from Google Drive `MyDrive/cits4012/e4_full/`:

- `e4_logs/a1_no_position/`, `e4_logs/a1_one_head/`, `e4_logs/a3_no_dropout/` (training runs): `config.json`, `m3_summary.csv`,
  `m3_model_table.csv`, `m3_selection.json`, `m3_curves.png`, and per seed `history.csv` and `val_predictions.csv`.
- `e4_logs/a2a_no_goal/seed4012/`, `seed4013/`, `seed4014/` (A2-a, evaluation only): `metrics.json` and `valid_predictions.csv`.

Model weights (`best.pt`) are on Google Drive only, not in the repository. The A2-a evaluation uses the
main model's checkpoints from E2 (`E2/checkpoints/`), not copies made in E4.

## Reproduce

1. Open `E4_ablation1 (2).ipynb` in Colab and select a T4 GPU.
2. Run all. Runs that already exist and pass the checks are reused, not retrained.
   A fresh run of one ablation (3 seeds) takes about 50 minutes on a T4.
3. The notebook stops with an error if a run folder is partial or fails a check.

## Provenance

- E2 code paths (`scripts/training`, `scripts/models`, `scripts/data_loader`, `scripts/evaluation`,
  `scripts/tokenizer`, `datasets`) are unchanged between d0268525 and the E4 commit 2fa9820cc.
  The notebook checks this with `git diff` before using any run.
- Each run's `config.json` records the base commit and the single changed setting.

## Caveats

1. Three seeds per experiment. Standard deviations and per-seed tests are rough. Read the direction
   across seeds together with the mean.
2. All ablations reuse the main model's hyperparameters without retuning, so each ablation is the
   main model with one change and no adaptation.
3. The main model was selected on this validation set, so its validation numbers are slightly optimistic
   relative to the ablations.
4. Results here are validation only. Test-set evaluation of the frozen main model and of the ablations
   comes later, once, as the plan requires.
