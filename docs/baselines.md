# 基线

分工里基线是志豪的部分。这里是 B0、B1、B2、B3。

所有基线用 `piqa_to_dataframe` 的同一份划分。`valid_df` 和 `test_df` 先 `reset_index(drop=True)`，预测表的 `row_id` 就是行号，可以和主模型逐题做 McNemar。指标用 `compute_metrics`。预测表的列和 `evaluate()` 一样：`row_id, label, logit_1, logit_2, prob_sol2, pred, correct`。

要做选择的只有 B2 的 sum / mean，只看验证集。测试集只评选定的那一种。

从仓库根目录运行。

# B0 规则

不训练。三个规则在验证集和测试集上都算。

| 规则 | 做法 |
| --- | --- |
| random | numpy 种子 4012 抽一次，先验证集再测试集。`prob_sol2` 记 0.5 |
| always_sol1 | 总选 sol1。准确率等于标签 0 的比例 |
| shorter | 选字符数更少的方案，一样长时选 sol1 |

`logit_1` / `logit_2` 用 0/1 表示选了哪一边。

```bash
python -m scripts.baselines.b0_trivial
```

输出在 `reports/baselines/b0_trivial/`：`{规则}_valid_predictions.csv`、`{规则}_test_predictions.csv`、`metrics.json`。

# B1 不带注意力的 BiLSTM

不新写模型。`RNNWithAttention` 在 `enable_attention=false` 时用最后一层的隐状态，`forward` 返回的 weights 是 `[None, None]`。

训练设置都在 `config/b1_lstm_noattn.json`：LSTM、AdamW（`weight_decay` 0.01）、学习率 0.001、batch 512、最多 150 轮、patience 30、种子 4012 / 4013 / 4014。要改协议就改这个文件。

```bash
python -m scripts.training.run_m3 --config config/b1_lstm_noattn.json
```

`run_m3.py` 不使用测试集。三个种子跑完之后，每个 `best.pt` 再用下面的评估脚本在测试集上各评一次。输出在 `reports/baselines/b1_lstm_noattn/`。

# B2 零样本语言模型

`openai-community/gpt2` 和 `Qwen/Qwen2.5-0.5B`（base，不是 Instruct）。用 `transformers` 和各自的分词器。不要用 `piqa_bpe_v3`。

输入固定为：

- `Question: {goal}\nAnswer:`
- 续写是 `" " + {方案}`

context 和 continuation 分开编码再拼起来。只对 continuation 的 token 求对数概率。每个方案有两种分数：token 对数概率的和（sum），以及和除以 token 数（mean）。验证集上准确率更高的那种作为该模型的主结果；打平用 mean。测试集只评这一种。

CPU 和 GPU 都用 float32。只差一个词的两个方案分数很接近，float16 的舍入可能把高低判反。GPU 按 batch 做右侧 padding，attention mask 把 pad 标成 0。CPU 的 float32 矩阵乘会随 batch 形状改变，所以 CPU 逐条前向，batch=1 和 batch=8 的分数相同。分数如果不是有限值，程序会报错，避免 NaN 被当成全选 sol1。

```bash
python -m scripts.baselines.b2_zero_shot_lm --models openai-community/gpt2 Qwen/Qwen2.5-0.5B --splits valid test
```

本地试跑可以加 `--limit 200 --splits valid --models openai-community/gpt2`。完整运行放到 Colab GPU：

```bash
python -m scripts.baselines.b2_zero_shot_lm \
    --models openai-community/gpt2 Qwen/Qwen2.5-0.5B \
    --splits valid test \
    --batch-size 8
```

输出在 `reports/baselines/b2_zero_shot_lm/<模型名>/`。`valid_sum_predictions.csv` 和 `valid_mean_predictions.csv` 是两种分数；`valid_predictions.csv` 和 `test_predictions.csv` 是选定的那种。`metrics.json` 里有两种验证集结果和最终选择。

GPT-2 约 1.24 亿参数，Qwen2.5-0.5B 约 5 亿，我们的模型约 100 万。差的是训练方式。两个模型的预训练数据都可能见过 PIQA。

# 加载 checkpoint

主模型、B1、消融，冻结之后都用这一个脚本评估。它读训练 config，按 `build_model` 建模型，再 `torch.load(..., weights_only=True)`。

```bash
python -m scripts.evaluation.evaluate_checkpoint \
    --config config/m3_params.json \
    --model transformer \
    --checkpoint path/to/best.pt \
    --split test \
    --output-dir reports/eval/run_name
```

`--split` 是 `valid` 或 `test`。`--no-include-goal` 只用于 A2-a：评估已经训练好的模型时去掉 goal，输入是 `[CLS] [SEP] 方案 [EOS]`。A2-b 是不带 goal 从头训练，训练输入是 `[CLS] 方案 [EOS]`，格式不同，不能用这个参数评估。默认保留 goal。

`build_model` 会读 `dropout_rate`（默认 0.1）、Transformer 的 `pooling`（默认 `cls`）和 `enable_pos_emb`（默认 true）。这些缺省和模型构造函数一样。评估 mean pooling 的 checkpoint 时，config 里必须写 `"pooling": "mean"`，否则权重能加载，结果是错的。

# B3 微调预训练编码器

`distilbert-base-uncased` 和 `roberta-base`，用 `AutoModelForMultipleChoice` 和各自的分词器。每题两个句对 `(goal, sol1)`、`(goal, sol2)`，最长 128 个 token，batch 内动态 padding，形状 `[B, 2, L]`。

训练是自己写的 PyTorch 循环。AdamW，学习率 2e-5，weight decay 0.01，前 10% 步数 warmup，之后线性衰减到 0。batch 16，3 轮，梯度裁剪 1.0。GPU 上训练用 fp16 混合精度，评估用 float32。CPU 上训练和评估都是 float32。

每轮结束在验证集上评估，按验证准确率留下最好的一轮，参数只存在内存里，训练结束后载回。测试集只评这一次。种子默认 4012 / 4013 / 4014，先固定种子再加载模型。某个 seed 失败会记在 `summary.json` 里，不重跑。

```bash
python -m scripts.baselines.b3_finetune \
    --models distilbert-base-uncased roberta-base \
    --seeds 4012 4013 4014 \
    --epochs 3 --lr 2e-5 --batch-size 16 --max-length 128
```

输出在 `reports/baselines/b3_finetune/<模型名>/seed<seed>/`：`history.csv`、`valid_predictions.csv`、`test_predictions.csv`、`metrics.json`。每个模型目录还有 `summary.json`（均值 ± 标准差）。不保存模型权重。`metrics.json` 里有句对超过 128 的截断比例。

DistilBERT 约 6600 万参数，RoBERTa-base 约 1.25 亿，大约是我们模型的 60 到 125 倍，而且经过预训练。预训练语料可能包含 PIQA 相关的文本。
