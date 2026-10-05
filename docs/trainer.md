# Introduction

训练循环已经封装为 `train()`，完成了 epoch 迭代、前向与反向传播、梯度范数记录、每轮验证、最佳 checkpoint 保存和训练日志落盘。

`train()` 不计算指标：验证指标调用 `scripts/evaluation/evaluate.py` 的 `evaluate()`，训练集指标调用同一文件的 `compute_metrics()`。

# Usage

```python
from scripts.training.trainer import train, set_seed
from scripts.evaluation.evaluate import evaluate, compute_metrics

params = json.load(open("config/m3_params.json"))
params["learning_rate"] = 1e-3

set_seed(params["seed"])            # 必须在建立 DataLoader 之前
train_loader, validate_loader, test_loader = get_piqa_dataloaders(
    base_path=BASE_PATH, tokenizer=tokenizer, batch_size=params["batch_size"])

history, summary = train(
    model, train_loader, validate_loader, params,
    evaluate_fn=evaluate,
    compute_metrics_fn=compute_metrics,
    test_dataloader=None,                  # 模型选择阶段不碰测试集
    model_name="transformer",
    out_dir="reports/m3_results",
    run_id="m3_transformer_lr0.001")
```

每个 epoch 打印一行，`* best` 表示这轮刷新了验证准确率：

```
Epochs: 1 | Train Loss:  0.693 | Train Accuracy:  0.512 | Val Loss:  0.692 | Val Accuracy:  0.519 | 0m 11s (- 10m 49s)  * best
```

# Parameters

|参数|说明|
|--|--|
|`model`|`RNNWithAttention` 或 `TransformerClassifier`，两种 forward 返回格式都兼容|
|`train_dataloader` / `val_dataloader`|`get_piqa_dataloaders()` 返回的前两个|
|`params`|超参字典，必须含 `learning_rate` 和 `epochs`，可选 `grad_clip`。原样写进 `summary.json`|
|`evaluate_fn`|`evaluate(model, loader, device) -> (metrics, predictions)`|
|`compute_metrics_fn`|`compute_metrics(labels, preds) -> dict`|
|`test_dataloader`|传入时训练结束后用最佳 checkpoint 评估一次。模型选择阶段传 `None`|
|`model_name`|`"rnn"` 或 `"transformer"`，写进 DataFrame 和文件名|
|`out_dir` / `run_id`|输出目录和文件名前缀|
|`verbose`|是否打印进度，默认 `True`|

# Returns

`history`：`pandas.DataFrame`，一个 epoch 一行，20 列。

|列|来源|
|--|--|
|`run_id`、`model`、`epoch`|trainer|
|`train_loss`|trainer，按「总 loss ÷ 题数」|
|`train_accuracy`、`train_precision`、`train_recall`、`train_f1`|`compute_metrics()`|
|`valid_loss`、`valid_accuracy`、`valid_precision`、`valid_recall`、`valid_f1`|`evaluate()`|
|`acc_gap`|`train_accuracy - valid_accuracy`，观察过拟合|
|`grad_norm_mean`、`grad_norm_max`|裁剪前的梯度范数|
|`train_seconds`、`valid_seconds`、`epoch_seconds`|本轮耗时|
|`is_best`|本轮是否刷新记录|

`summary`：`dict`，含 `config`、`n_params`、`n_params_no_embedding`、`best_epoch`、`best_valid_accuracy`、`checkpoint` 路径。传入 `test_dataloader` 时多一个 `test` 键。

# Output Files

|文件|内容|
|--|--|
|`<run_id>_history.csv`|上面那张 DataFrame|
|`<run_id>_summary.json`|上面那个 dict|
|`<run_id>_best.pt`|验证准确率最高那轮的 `state_dict`|
|`<run_id>_best_predictions.csv`|同一轮的逐题预测|
|`<run_id>_test_predictions.csv`|仅在传入 `test_dataloader` 时生成|

加载 checkpoint 时模型的构造参数必须和训练时一致，这些参数在 `summary.json` 的 `config` 里。

# Experiment Notebook

`notebooks/M3_selection.ipynb` 是 M3 的实验入口，一次运行完成 2 个模型 × 2 个学习率共 4 次训练，并按验证准确率选出主模型。

数据划分和 Tokenizer 由 cell 7 开头的一个开关控制：

```python
SPLIT_MODE = "row"          # "group" or "row"
```

|`SPLIT_MODE`|划分|Tokenizer|
|--|--|--|
|`"row"`|仓库默认的按行随机划分|`piqa_bpe_v2.model`|
|`"group"`|按 goal 分组，同一个 goal 的所有行不拆开|在新训练子集上重训的 `piqa_bpe_m3.model`|

两者必须配套，不能混用：`piqa_bpe_v2.model` 是在按行划分的训练子集上训练的，配合按 goal 划分会让约 89% 的新验证文本进入过 Tokenizer 的训练语料。开关同时控制这两件事，因此不会出现错配。

按行划分会让同一个 goal 同时出现在训练集和验证集，实测 157 行（9.7%）属于这种情况。按 goal 划分可以消除这一问题，训练集和验证集的大小不变（14502 / 1611）。

## 切换后如何确认实际生效

`"group"` 分支的实现方式是改写 `scripts/data_loader/load_piqa_data.py`。cell 7 开头会先把该文件还原成仓库版本并清除已缓存的模块，再按需打补丁，因此**切换后直接重跑即可，不需要删除 runtime**。

运行 cell 7 后通过输出确认实际生效的划分：

```
"row"    → validation rows whose goal appears in train: 157 (9.7%)
           validation label counts: {0: 836, 1: 775}

"group"  → validation rows whose goal appears in train: 0 (0.0%)
           validation label counts: {0: 808, 1: 803}
```

cell 9 会打印实际加载的 Tokenizer，`"row"` 为 `using the committed piqa_bpe_v2.model`，`"group"` 为 `retrained the tokenizer on the new training subset (43506 lines)`。

## 输出

结果写入 `reports/m3_results/`，最后打包为 ZIP 下载，内容包括 4 个 run 各自的 `history.csv`、`summary.json`、`best.pt`、`best_predictions.csv`，以及汇总的 `m3_all_history.csv`、`m3_selection.json`、`m3_params.json`、`m3_curves.png` 和本次使用的 Tokenizer。

`.pt` 被 `.gitignore` 排除，不会进入版本库，需要时通过 ZIP 单独传递。

# Note

- `set_seed()` 必须在建立 DataLoader 之前调用。训练 DataLoader 的 batch 顺序来自全局 torch RNG，之后再设种子不会改变顺序。
- `evaluate()` 内部自己处理 `model.eval()` 并恢复原模式，在训练循环中调用是安全的。
- 训练集指标用训练过程中收集的预测计算，不要对 `train_loader` 调用 `evaluate()`，那会多跑一遍完整前向。
- `is_best` 标记的是「刷新记录」而不是「全局最优」，第 1 轮必然为 `True`。取最优轮应该用 `history["valid_accuracy"].idxmax()` 或 `summary["best_epoch"]`。
