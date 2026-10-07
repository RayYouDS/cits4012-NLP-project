# Introduction

训练循环在 `scripts/training/trainer.py`。它更新调用方传入的模型，并把验证准确率最高那一轮的参数载回这个模型对象。

优化器、随机种子和写盘在调用方。损失函数在 `train()` 内部固定为 `nn.CrossEntropyLoss()`。`scripts/training/run_m3.py` 是 M3 的入口。

验证指标用 `scripts/evaluation/evaluate.py` 的 `evaluate()`。训练集上用于画过拟合的准确率也用同一次 `evaluate()`，在 `eval()` 模式下计算，这样 Transformer 的 dropout 不会和没有 dropout 的 RNN 比错。

# Usage

从仓库根目录运行：

```bash
python -m scripts.training.run_m3 --config config/m3_params.json
```

在自己的脚本里调用训练循环：

```python
from scripts.training.trainer import set_seed, train
from scripts.training.bucket_sampler import LengthBucketBatchSampler, piqa_lengths
from scripts.evaluation.evaluate import evaluate

set_seed(4012)   # 建模型之前
model = build_model(...).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

generator = torch.Generator()
generator.manual_seed(4012)
train_loader = DataLoader(
    PIQADataset(train_df),
    batch_sampler=LengthBucketBatchSampler(piqa_lengths(train_df), batch_size=64, generator=generator),
    collate_fn=PIQACollator(tokenizer),
)

model, history = train(
    model, train_loader, val_loader, optimizer, device,
    max_epochs=1000, patience=None, train_eval_loader=train_eval_loader,
)
torch.save(model.state_dict(), "best.pt")
```

`train()` 返回时，`model` 里已经是验证准确率最高那一轮的参数。调用方再决定要不要写盘。

# Parameters

|参数|说明|
|--|--|
|`model`|已经放到 `device` 上的 `RNNWithAttention` 或 `TransformerClassifier`|
|`train_loader`|训练集。batch 顺序由 sampler 自己的 `torch.Generator` 决定|
|`val_loader`|验证集，`shuffle=False`|
|`optimizer`|调用方创建。M3 的 `optimizer` 取 `adam` / `adamw` / `sgd`。`weight_decay` 为 `null` 时用该优化器的 PyTorch 默认值。`momentum` 只对 SGD 生效，默认 0.9|
|`device`|`"cpu"` 或 `"cuda"`|
|`max_epochs`|最多训练轮数。当前 M3 配置是 1000，用来做 GPU 压力测试|
|`patience`|传入整数才早停：验证准确率连续这么多轮没有刷新就停止。`None` 时跑满 `max_epochs`。当前配置是 `null`|
|`grad_clip`|梯度裁剪上限。`None` 时只记录范数，不裁剪|
|`train_eval_loader`|可选。传了就每轮用 `evaluate()` 在这个固定子集上记 `train_eval_accuracy`|
|`verbose`|是否打印每轮一行|

`set_seed(seed=4012)` 固定 Python、NumPy、PyTorch 和 CUDA 的种子，在创建模型之前调用。它不负责训练集的 batch 顺序。那个顺序要靠传给 `DataLoader` 或 `LengthBucketBatchSampler` 的 `torch.Generator`。

# Returns

`(model, history)`。

`history` 每个 epoch 一行：

|列|说明|
|--|--|
|`epoch`|从 1 开始|
|`train_loss`|训练模式，总 loss / 题数|
|`train_accuracy`|训练模式的 argmax。Transformer 含 dropout，不能拿来和 RNN 比过拟合|
|`train_eval_accuracy`|`train_eval_loader` 上、eval 模式的准确率。没传 loader 时为空|
|`valid_loss` / `valid_accuracy` / `valid_precision` / `valid_recall` / `valid_f1`|`evaluate()`|
|`grad_norm_mean` / `grad_norm_max`|裁剪前的梯度总范数|
|`is_best`|这一轮是否刷新验证准确率。第一轮一定是 `True`。真正的最好轮用 `valid_accuracy` 最大的那一行|
|`epoch_seconds`|这一轮训练加验证的耗时|

# 速度

每道题平均大约 35 个 token，但长度差很大。在按 goal 切分的这版训练集上，`batch_size=64` 随机打乱后的平均补齐长度是 135.6；`LengthBucketBatchSampler` 先打乱，再在 `batch_size * 50` 的窗口里按长度排序、切成 batch，然后打乱 batch 的顺序，平均补齐长度是 40.1。

另外三件事让 GPU 不用每个 batch 都停下来：

- loss 和正确数在 device 上累加，每个 epoch 只同步一次
- 最好一轮的 `state_dict` 留在内存里，训练结束才由调用方写一次盘
- `num_workers > 0` 时打开 `persistent_workers`，CUDA 上打开 `pin_memory`

数据每个 batch 都会 `.to(device)`。设备和模型不一致时 PyTorch 会直接报错。

验证集和测试集仍然 `shuffle=False`，不用分桶，这样 `evaluate()` 的 `row_id` 稳定。

# M3

`config/m3_params.json` 里是三组模型（`gru`、`lstm`、`transformer`）、两个学习率、种子 `[4012]`、`max_epochs=1000`、`patience=null`。默认跑满 1000 轮，不早停；最好一轮的参数仍然只留在内存里，训练结束时载回模型，再由 `run_m3.py` 写盘。划分和 tokenizer 都沿用仓库里的 `tokenize_piqa_dataframe` 与 `piqa_bpe_v3.model`。不要在训练脚本里重训 tokenizer，也不要改切分。

每个 run 的名字是 `m3_{model}_lr{lr}_seed{seed}`。输出在 `reports/m3_results/`：

|文件|内容|
|--|--|
|`<run_id>_best.pt`|该 run 验证准确率最高一轮的 `state_dict`|
|`<run_id>_history.csv`|该 run 的 history|
|`<run_id>_val_predictions.csv`|最好一轮在验证集上的逐题预测|
|`m3_summary.csv`|每个 run 一行，含最好轮、跑了多少轮、验证指标、eval 模式差距、耗时、参数量|
|`m3_model_table.csv`|每个模型按各 seed 的平均验证准确率选出的学习率，以及均值和标准差|
|`m3_selection.json`|平均验证准确率前两名，用 `seeds` 里的第一个种子做 McNemar 检验|
|`m3_curves.png`|左图每轮验证准确率，右图 `train_eval_accuracy - valid_accuracy`|

McNemar 看的是同一批验证题上谁做对了。`p < 0.05` 才把差别当成显著。准确率差一个点、但分歧接近对半时，不应当成模型胜负。

# Note

- 比较过拟合用 `train_eval_accuracy - valid_accuracy`。`train_accuracy` 是训练模式下的数字，Transformer 默认 dropout 0.1，RNN 没有 dropout。
- 早停只有在 `patience` 传入整数时才会发生。不管有没有提前停止，返回的模型都等于 history 里验证准确率最高的那一轮。
- 测试集不参与 M3。
