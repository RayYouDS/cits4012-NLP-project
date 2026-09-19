# 概述

实验对四个数据集进行了初步分析：

| 数据集     | 大小 (MB) | 优点                                                                      | 缺点                                                                                                                                 |
| ------- | ------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| PIQA    | 4.68    | - 高度结构化数据，问题+回答简短，几乎不需要预处理<br>- 二选一问题 -> 二分类模型<br>- 记录充足：训练集有 16113 条记录 | - 问题简短，语料可能不足<br>- 选项间相似度高，有时只有一个单词的差异                                                                                             |
| MultiRC | 3.86    | - 数据结构为 Paragraph -> Questions -> Answers 的嵌套结构，很接近实际的阅读理解任务            | - 多层 JSON 嵌套，解析困难<br>- 问题包含 html 标签，需要额外清洗步骤<br>- 存在一个问题对应多个正确选项的情况 -> 需要类似 SQL Query 的机制<br>- Paragraph 只有 456 个，因此相同不同语境的训练集相对较少 |
| ReClor  | 5.25    | - 每个问题信息充足，包含背景信息 (Context 字段)<br>- 四选一问题 -> 四分类模型                      | - 训练集只有 4638 条记录                                                                                                                   |
| TweetQA | 2.81    | - 结构化 CSV 数据                                                            | - 非选择 QA 模型，需要 Seq 2 Seq 架构，技术难度高<br>- 包含非 UTF-8 字符<br>- 数据量最小                                                                     |

# 初步结论

PIQA 的优势非常明显：

- 二分类问题，可以使得模型的输入和输出都较为简单
- 基本不用花大量时间处理数据
- 训练数据相对充足，不容易发生过拟合

但最大问题也来自于此：

- 任务太过简单，可能做不出特别有深度的研究
- Attention 的研究空间相对有限，可能很难解释为什么 Attention 在这个任务中有效

# PIQA 数据集分析

PIQA Train 的前十条记录如下：

|     | goal                                                          | sol1                                                                                                                                                                          | sol2                                                                                                                                                                            | label |
| --- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| 0   | When boiling butter, when it's ready, you can                 | Pour it onto a plate                                                                                                                                                          | Pour it into a jar                                                                                                                                                              | 1     |
| 1   | To permanently attach metal legs to a chair, you can          | Weld the metal together to get it to stay firmly in place                                                                                                                     | Nail the metal together to get it to stay firmly in place                                                                                                                       | 0     |
| 2   | how do you indent something?                                  | leave a space before starting the writing                                                                                                                                     | press the spacebar                                                                                                                                                              | 0     |
| 3   | how do you shake something?                                   | move it up and down and side to side quickly.                                                                                                                                 | stir it very quickly.                                                                                                                                                           | 0     |
| 4   | Clean tires                                                   | Pour water, cape off caked on dirt. Use  speed wool to clean out crevices and sparrow spaces.                                                                                 | Pour water, scrape off caked on dirt. Use a steel wool to clean out crevices and narrow spaces.                                                                                 | 1     |
| 5   | how do you taste something?                                   | smell it enough to taste it.                                                                                                                                                  | place it in your mouth to taste.                                                                                                                                                | 1     |
| 6   | To create a makeshift ice pack,                               | take a sponge and soak it in oil. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack. | take a sponge and soak it in water. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack. | 1     |
| 7   | What should I use as a stain on a wooden bowl I've just made. | You should coat the wooden bowl with a butcher block oil & finish per manufacturer directions.                                                                                | You should coat the wooden bowl with a butcher knife oil & finish per manufacturer directions.                                                                                  | 0     |
| 8   | How to boil eggs.                                             | Place your eggs in a pot and cover with no water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.            | Place your eggs in a pot and cover with cold water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.            | 1     |
| 9   | how do you stab something?                                    | stick a sharp object through it.                                                                                                                                              | pin it with a sharp object.                                                                                                                                                     | 0     |

可以观察到：PIQA 的关键不在长文本噪声，而在非常细微的常识差异：oil vs water、butcher block oil vs butcher knife oil、no water vs cold water。因此预处理要“修格式、不改词义”，尤其是在每一个训练语料简短的情况下，不要让预处理破坏语义。

对 PIQA 文本，不可以采用的预处理有：

- Stopwords removal：会损失 in、with、by、no、to 等关系词；第 8 条的 no water 就是典型反例。
- 删除标点：medium-high、ziploc bag、问号等表达应保留，且 `SentencePiece` 等分词器可以很好地处理标点
- Case-Folding：使用 `SentencePieces` 等 Subword 分词器时，不建议将文本全部转为小写
- stemming / lemmatization：可能会损害语义，在每个语料较文简短的情况下不建议

可以采用的预处理方案：

- Normalization：文本存在多个连续空格的情况，可以进行压缩处理

