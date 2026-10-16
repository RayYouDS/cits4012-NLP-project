# E4: Ablation experiments on the main model (M6)

This folder holds the E4 training logs, the result tables, the figure, the environment record and the
notebook that checks them. All numbers are on the **validation set**. The test set is not loaded or evaluated here.

## Reference model

The main model is the frozen E2 configuration (`experiments/E2/results/final_3seed/config.json`, E2 `_commit` d0268525):
Transformer, 2 layers, 4 heads, embedding dim 100, dropout 0.3, AdamW lr 9e-4, weight decay 0.01,
batch 512, 300 epochs, no early stopping, 933,357 parameters.
Its validation accuracy over seeds 4012, 4013, 4014 is 0.5940 ± 0.0081 (at the best epoch).

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
- B1 (no-attention baseline): owned by a teammate and reported in the baselines section. It is not part of E4.

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
The no-dropout run also stops improving early, so its accuracy gap mixes the effect of dropout with the
effect of training length.

A2-a: with goal removed, 302, 268 and 249 of 1,606 predictions change (seeds 4012, 4013, 4014).
The changes go both ways, but more answers are lost without the goal than gained.
The model was never trained without the goal, so this measures sensitivity to input change, not
reasoning without the goal.

## Environment

The environment of the original runs is in `env.json` and `env_freeze.txt` (Colab).

| Item | Value |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| GPU | Tesla T4 |
| E4 code commit (`baselines`) | 2fa9820ccd9e0c8833ffbf6ec900c5d7f0f7cd94 |
| E2 code commit (main model) | d0268525b83d7af9f5d5b3a684b6ad8dec00d4d7 |

## Files

Top level of this folder (`experiments/E4 updated/`):

- `E4_ablation1 (2).ipynb`: the Colab notebook that produced these runs (training, A2-a, tables, figure, environment record).
- `E4_section2_final.ipynb`: the Section 2 notebook for markers. It reads the saved logs in `experiments/E2/results/final_3seed/` and `experiments/E4 updated/e4_logs/`, checks them, and writes `training_summary.csv` and `a3_gap_loss.png` here. No GPU, Drive or internet is needed. The ablation result tables below (Table 1, Table 2 and the A2-a summary) are the files from the original Colab run; the Section 3 notebook builds the comparison tables.
- `README4.md`: this file.
- `training_summary.csv` (written by Section 2): per-run means and s.d. (valid accuracy, gaps, validation loss, best epoch).
- `e4_table1.csv`: the same per-experiment summary from the original Colab run.
- `e4_table2.csv`: paired comparison with each ablation (Δ, per-seed McNemar p). Section 3.
- `e4_a2a_summary.csv`: A2-a mean and s.d. from the original Colab run.
- `a3_gap_loss.png`: validation loss and overfitting gap curves, main vs A3, per seed (written by Section 2).
- `valid_accuracy_comparison.png`: validation accuracy of the main model and each ablation, mean ± s.d. over 3 seeds (written by Section 2).
- `env.json`, `env_freeze.txt`: runtime and package versions of the original runs.

Run logs (`e4_logs/`):

- `e4_logs/a1_no_position/`, `e4_logs/a1_one_head/`, `e4_logs/a3_no_dropout/` (training runs): `config.json`, `m3_summary.csv`,
  `m3_model_table.csv`, `m3_selection.json`, `m3_curves.png`, and per seed `history.csv` and `val_predictions.csv`.
- `e4_logs/a2a_no_goal/seed4012/`, `seed4013/`, `seed4014/` (A2-a, evaluation only): `metrics.json` and `valid_predictions.csv`.

Model weights:

- The main model's checkpoints (3 × `best.pt` state_dict, about 4.6 MB each) are in `experiments/E2/checkpoints/`. A2-a was evaluated on these. They can be loaded without retraining.
- The ablation checkpoints (`a1_no_position`, `a1_one_head`, `a3_no_dropout`; 3 seeds each) were produced on Colab and are **not included** in this repository. The results here come from the saved logs and validation predictions, and the checks in the notebook do not need the weights.

## Reproduce

1. Download the repository and open `experiments/E4 updated/E4_section2_final.ipynb` (Jupyter or Colab, CPU is enough).
2. Run all cells from the repository root. The default mode reads the saved logs, so nothing is retrained. Section 2.2b loads the saved main-model weights (`experiments/E2/checkpoints/`, seed 4012) and checks the trainable parameter count (933,357).
3. The notebook stops with an error if a run folder is missing a file or fails a check.

To retrain on Colab with a T4 GPU, set `RUN_TRAINING = True` in the configuration cell. A fresh run of one ablation
(3 seeds) takes about 50 minutes. Runs that already exist and pass the checks are reused, not retrained.

## Provenance

- E2 code paths (`scripts/training`, `scripts/models`, `scripts/data_loader`, `scripts/evaluation`,
  `scripts/tokenizer`, `datasets`) were checked unchanged between d0268525 and the E4 commit 2fa9820cc
  when the Colab notebook ran (`git diff`). The Section 2 notebook repeats this check only when the copy has a `.git` folder.
- Each run's `config.json` records the base commit (`_base_commit`) and the single changed setting (`_changes`).

## Caveats

1. Three seeds per experiment. Standard deviations and per-seed tests are rough. Read the direction
   across seeds together with the mean.
2. All ablations reuse the main model's hyperparameters without retuning, so each ablation is the
   main model with one change and no adaptation.
3. The main model was selected on this validation set, so its validation numbers are slightly optimistic
   relative to the ablations.
4. Results here are validation only. Test-set evaluation of the frozen main model and of the ablations
   comes later, once, as the plan requires.
5. The plan (section 0) asks for `best.pt` for every run. The ablation checkpoints are not in this repository, so markers cannot reload those models. The main model's checkpoints are included.
