# E5 注意力分析

主模型是 E2 定稿的 Transformer：2 层、4 头、embedding 100、CLS pooling、post-norm、dropout 0.3。
权重是 `experiments/E2/checkpoints/` 里三个 seed 的 `best.pt`。全部数字来自验证集 1606 题，没有使用测试集。
分类只用位置 0 的 [CLS]，下面的注意力都是 [CLS] 这一行。rollout 按 Abnar and Zuidema (2020)，先对头取平均。

注意力分布和做对做错同时出现，只能说明相关，不能说明模型是因为看到这些 token 才做出选择（Jain and Wallace, 2019; Wiegreffe and Pinter, 2019）。

## 验证

- seed 4012: [CLS] 和 `encode()` 的最大绝对误差 7.153e-07；有效 query 行在有效 token 上的和落在 [0.99999976, 1.00000024]，padding 键上的最大值 0.000e+00；pred / label / correct 和 `val_predictions.csv` 一致，logit 与概率的最大绝对差 1.458e-06。
- seed 4013: [CLS] 和 `encode()` 的最大绝对误差 8.345e-07；有效 query 行在有效 token 上的和落在 [0.99999976, 1.00000024]，padding 键上的最大值 0.000e+00；pred / label / correct 和 `val_predictions.csv` 一致，logit 与概率的最大绝对差 1.409e-06。
- seed 4014: [CLS] 和 `encode()` 的最大绝对误差 8.345e-07；有效 query 行在有效 token 上的和落在 [0.99999976, 1.00000024]，padding 键上的最大值 0.000e+00；pred / label / correct 和 `val_predictions.csv` 一致，logit 与概率的最大绝对差 1.457e-06。

## goal、方案、特殊符号

每个数是三个 seed 的均值 ± 样本标准差。一道题的两个候选各算一次，再对全部验证题取平均。token 占比是这三组的长度占比。

| 区域 | token 占比 | rollout | 最后一层 |
| --- | --- | --- | --- |
| goal | 0.294 ± 0.000 | 0.150 ± 0.013 | 0.213 ± 0.013 |
| solution | 0.581 ± 0.000 | 0.579 ± 0.016 | 0.724 ± 0.032 |
| special | 0.125 ± 0.000 | 0.271 ± 0.003 | 0.064 ± 0.018 |

E4 的 A2-a 在评估时去掉 goal，验证准确率下降约 2.4 个点，三个 seed 里改变的预测大约是 15.5%–18.8%（`experiments/E4/README4.md`）。
goal 的 rollout 是 0.150 ± 0.013，最后一层是 0.213 ± 0.013，都低于 token 占比 0.294 ± 0.000。方案的 rollout 是 0.579 ± 0.016，接近 token 占比 0.581 ± 0.000；最后一层是 0.724 ± 0.032，高于长度占比。特殊符号的 rollout 是 0.271 ± 0.003，高于 token 占比 0.125 ± 0.000；最后一层只有 0.064 ± 0.018。
rollout 在每一层用 A' = 0.5·A + 0.5·I，单位阵会把一部分质量留在 [CLS] 上，所以特殊符号这一格偏高。最后一层没有加单位阵，特殊符号低于它的长度占比，方案则高于长度占比。
goal 在两种算法里都低于自己的长度占比。这和 A2-a 里大约 15%–19% 的预测发生改变是同一方向的相关。注意力权重不是因果解释。

## 差异词

两个方案按空格分词后做词级编辑距离。换成一个词记距离 1。距离不超过 2 的题里，把不同的词映射到以 ▁ 开头的 SentencePiece 词，再看 rollout 落在这些子词上的比例。
每题先对两个候选取平均，再在做对和做错里分开。括号里是该 seed 的 95% bootstrap 置信区间（2000 次百分位；Generator 种子是 4012 + seed）。跨 seed 再报均值 ± 标准差。

- 做对（每 seed 约 566 题）：rollout 0.116 ± 0.004，token 占比 0.074 ± 0.001。
  - seed 4012: rollout 0.114 [0.106, 0.122]，token 0.074 [0.070, 0.079]，n=564
  - seed 4013: rollout 0.113 [0.105, 0.120]，token 0.073 [0.069, 0.078]，n=565
  - seed 4014: rollout 0.121 [0.113, 0.128]，token 0.075 [0.071, 0.080]，n=570
- 做错（每 seed 约 392 题）：rollout 0.120 ± 0.003，token 占比 0.081 ± 0.002。
  - seed 4012: rollout 0.117 [0.107, 0.128]，token 0.081 [0.075, 0.087]，n=395
  - seed 4013: rollout 0.120 [0.110, 0.130]，token 0.083 [0.077, 0.088]，n=394
  - seed 4014: rollout 0.123 [0.114, 0.134]，token 0.079 [0.074, 0.086]，n=389

三个 seed 上，做对和做错的 rollout 置信区间都有重叠。做对的均值是 0.116 ± 0.004，做错是 0.120 ± 0.003，两边都高于各自的 token 占比。

## 错误分析

| 分组 | 组 | 题数 | 准确率 |
| --- | --- | --- | --- |
| n_diff | 1 | 568 | 0.606 ± 0.016 |
| n_diff | 2 | 391 | 0.568 ± 0.016 |
| n_diff | 3-5 | 334 | 0.621 ± 0.030 |
| n_diff | >5 | 313 | 0.576 ± 0.024 |
| length | 5–20 | 410 | 0.587 ± 0.007 |
| length | 20–29 | 396 | 0.600 ± 0.012 |
| length | 29–43 | 404 | 0.621 ± 0.013 |
| length | 43–498 | 396 | 0.567 ± 0.015 |

准确率没有随着词级距离变大而下降：距离 1 为 0.606 ± 0.016，距离 2 为 0.568 ± 0.016，距离 3-5 为 0.621 ± 0.030，距离 >5 为 0.576 ± 0.024。其中 2 最低，3-5 最高，误差条有重叠。
序列长度按两个候选里较长的一段取四分位，从短到长是 5–20（n=410，0.587 ± 0.007），20–29（n=396，0.600 ± 0.012），29–43（n=404，0.621 ± 0.013），43–498（n=396，0.567 ± 0.015）。最长一组的右端拉得很长，因为验证集长度右偏。这一组低于中间的组。

## 删词（可选检查）

只改正确答案那一侧的方案：删掉 rollout 最高的词，或者随机删一个词（20 次取平均）。数字是正确选项概率的变化，三个 seed 各自的平均值再汇总。
- 删最高词：-0.040 ± 0.017
- 随机删一个词：-0.007 ± 0.001
- 最高词本身就是差异词的比例：0.398 ± 0.006
删掉 rollout 最高的词时，正确选项概率下降得比随机删词更多。
只改了正确答案那一侧的方案，另一侧保持原样。最高词里不到一半本身就是差异词，所以多数时候删掉的不是两个方案的对比词。
概率变了，说明这些词和这次输出有关。这仍然不是注意力导致了选择的证明（Jain and Wallace, 2019; Wiegreffe and Pinter, 2019）。

## 例子

图只使用 seed 4012。先限制两个候选都不超过 30 个 token，再按 |P(sol2) − 0.5| 从高到低取，并列时取 row_id 小的。

- row 1562（把握最大且做对）：What is an easy way to keep a room dark during the day?
  正确答案 sol1，预测 sol1，P(sol2)=4.19e-05。差异词是「Purchase / blackout / curtains / liners.」对「Paint / your / windows / white / paint.」。
  两个候选平均后，rollout 在 goal 0.336、方案 0.391、特殊符号 0.273；差异词的 rollout 0.378，token 占比 0.250。
  词级距离是 5，两个方案大部分词都不同。差异词 rollout 0.378，token 占比 0.250，这个比例铺在整个方案上。
  这道题的 goal rollout 0.336，低于它自己的 token 占比 0.583，也高于验证集 goal rollout 的平均 0.150。
  最后一层四个头的最大值不在同一个子词上。sol1：头1「liners」，头2「Purchase」，头3「Purchase」，头4「liners」（SEP 左侧权重之和平均 0.46，摊在多个子词上） sol2：头1「Paint」，头2「paint」，头3「Paint」，头4「paint」（SEP 左侧权重之和平均 0.47，摊在多个子词上）。
- row 165（把握最大且做对）：To report a stolen vehicle to the police,
  正确答案 sol2，预测 sol2，P(sol2)=0.9999。差异词是「emergency / phone / number / ask / for / help.」对「non-emergency / line / report / it / there.」。
  两个候选平均后，rollout 在 goal 0.161、方案 0.569、特殊符号 0.270；差异词的 rollout 0.348，token 占比 0.319。
  词级距离是 6，两个方案大部分词都不同。差异词 rollout 0.348，token 占比 0.319，这个比例铺在整个方案上。
  这道题的 goal rollout 0.161，低于它自己的 token 占比 0.341。
- row 106（把握最大但做错）：handkerchief
  正确答案 sol1，预测 sol2，P(sol2)=0.999986。差异词是「washed / the / grease / off / the / pan / surface」对「played / checkers / with / pan / at / the / casino」。
  两个候选平均后，rollout 在 goal 0.110、方案 0.591、特殊符号 0.300；差异词的 rollout 0.591，token 占比 0.582。
  goal 在序列里只有 3 个子词。
  词级距离是 7，两个方案大部分词都不同。差异词 rollout 0.591，token 占比 0.582，这个比例铺在整个方案上。
- row 1135（把握最大但做错）：measuring cup
  正确答案 sol2，预测 sol1，P(sol2)=5.23e-05。差异词是「baby / wipe」对「stool」。
  两个候选平均后，rollout 在 goal 0.135、方案 0.550、特殊符号 0.315；差异词的 rollout 0.367，token 占比 0.141。
  goal 在序列里只有 2 个子词。
  两个方案不同的词，rollout 0.367，token 占比 0.141，高于长度占比。
- row 1142（最小差异且做对）：lid
  正确答案 sol2，预测 sol2，P(sol2)=0.9999。差异词是「bathtub」对「Tupperware」。
  两个候选平均后，rollout 在 goal 0.099、方案 0.581、特殊符号 0.320；差异词的 rollout 0.364，token 占比 0.156。
  goal 在序列里只有 1 个子词。
  两个方案不同的词，rollout 0.364，token 占比 0.156，高于长度占比。
- row 1296（最小差异但做错）：To add detail in paint without a brush, you can
  正确答案 sol2，预测 sol1，P(sol2)=0.0003。差异词是「tablespoon」对「toothpick」。
  两个候选平均后，rollout 在 goal 0.427、方案 0.279、特殊符号 0.294；差异词的 rollout 0.238，token 占比 0.059。
  两个方案不同的词，rollout 0.238，token 占比 0.059，高于长度占比。
  这道题的 goal rollout 0.427，低于它自己的 token 占比 0.647，也高于验证集 goal rollout 的平均 0.150。

## 图

- `figA_examples`：把握最大的做对、做错各 1 题，加上最小差异对的做对和做错。热图，较高。
- `figA_examples_all6`：上面的规则选出的全部 6 题。
- `figA_examples_compact`：报告用，双栏 6.3 英寸。同样 4 题，文字高亮，只画 rollout。
- `figA_examples_single_compact`：单栏 3.0 英寸。只放 row 1142 和 1296。
- `figB_share_and_accuracy`：区域占比，以及准确率随词级距离的变化。双栏。
- `figB_share_and_accuracy_compact`：报告用，单栏 3.0 英寸，两个小图上下排。
- `fig_heads_last_layer`：附图。把握最大且做对的那一题，最后一层 4 个头。
