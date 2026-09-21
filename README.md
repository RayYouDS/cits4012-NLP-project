# Latest Update

- Dataset 实验报告 [Dataset Inspection Report](reports/2026-09-19_dataset_inspection_report.md)
- [Tokenizer APi Reference](docs/tokenizer.md)
- [Get Data Loader API Reference](docs/get_data_loader.md)

# Assignment 简述

- Due: 2026-10-18 23:59

- 任务目标：设计、训练 QA 模型

- 架构：RNN/LSTM/GRU/Transformer, 必须含有注意力机制

- 定量分析：Precision, Recall, Accuracy, F1-Score, etc.

- 消融实验 (Controlled Ablation Experiment)：检查模型的组件对模型表现是否有帮助

- 定性分析：注意力热图 (heat map, etc) 等

- 基准实验：Random guess, Pretrained Checkpoints, etc

# 里程碑

| No  | Target                                 | Content                                                                                                       |
| --- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| M0  | Specifications  | 协作规范、数据集选择等 |
| M1  | Data Pipeline  | Raw data -> Dataloader, incl.<br>- Preprocessing<br>- Tokenizer |
| M2  | Minimum Model  | 设计一个可以跑通的最小模型   |
| M3  | Model Selection  | 选择一个模表现最好的模型作为主模型   |
| M4  | Main Exp. and baseline | 对主模型进行基准测试和hyperparameter grid search  |
| M5  | Fix model architecture and hypermeters | 确定最终模型架构和 Hyperparameters，并冻结主要实验设置 |
| M6  | Ablation and Qualitative Analysis      | 消融实验：移除注意力、candidate comparison/gate 等，验证模型组件对模型表现的影响，绘制注意力热力图，给出成功、失败案例  |
| M7  | Report and Reproduce  | 撰写报告、引用文献、转写为 Jupyter Notebook、在 Colab 上进行复现测试  |
| M8  | Final Check Metting  | 冻结报告修改，团队成员交叉检查最终提交材料 |


# 协作规范

- 为防止进度阻塞，原则上每个主要功能都应有两人共同参与设计或验证
- 涉及随机过程的代码，应设置随机数种子为 **4012**，以最大化保证可复现性
- 除了局部的对比实验以外，实验代码以 `.py` 脚本文件为主，而不是直接以 Notebook 作为主要开发环境，便于导入模块和版本控制。尤其是对于模型的 Class 架构类，应以将每一个实验 Class 保存为一个 `.py` 文件
- 对于具有明显架构差异的实验，可以使用具有描述性的文件名，比如 `bigru_attention.py`, `transformer_attention.py`，方便检查迭代轨迹
- 对于模块的使用说明文档，应集中保存在 `/docs` 目录下
- 阶段性实验报告以 Markdown 的格式撰写，保存在 `/reports` 目录下，实验报告可以是：
  - 数据检视和清洗实验
  - 模型对比实验
  - 消融实验
  - 基准测试
  - 其他值得注意的研究记录
- 实验过程中生成的图，应保存在 `/reports/figures` 目录下，方便后续 Latex 脚本引用
- main 分支为通过验证的主要版本，其中的代码应保证可以直接运行，因此除了初始化，禁止直接向 `main` 里 push
- 新功能、新实验和修复应在独立的 Branch 里完成，测试稳定后提交 PR 加入主代码库。分支可以设置：
  - data-pipeline
  - model-and-training
  - evaluation
  - ablation
  - baseline
  - report (latex 分支)
  - 其他功能明确的分支

# Standard Workflow

开始一个新任务时：

```bash
git checkout main
git pull origin main

git checkout -b branch_name
```

进行修改后：

```bash
git add .
git commit -m "commit message"
git push origin branch_name
```

每个主要功能完成后，应通过 Pull Request 合并到 main。

# References

