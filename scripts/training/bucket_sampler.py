"""按长度分桶的训练集 batch 采样。

PIQA 每题大约 35 个 token。batch 里只要混进一条长句，整批都要补到那个长度。
按 goal 切分的训练集上，batch_size=64 随机打乱的平均补齐长度是 135.6；先在一个窗口里按长度排好再切 batch，
平均补齐长度是 40.1。验证集和测试集不要用这个 sampler。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Sampler


def piqa_lengths(df):
    """每道题补齐后要占的长度：goal + 较长的那个 solution + [CLS]/[SEP]/[EOS]。"""
    goal = df["goal_tokens"].map(len).to_numpy()
    sol1 = df["sol1_tokens"].map(len).to_numpy()
    sol2 = df["sol2_tokens"].map(len).to_numpy()
    return (goal + np.maximum(sol1, sol2) + 3).astype(int).tolist()


class LengthBucketBatchSampler(Sampler):
    """每个 epoch：打乱下标，窗口内按长度排序，切成 batch，再打乱 batch 顺序。"""

    def __init__(self, lengths, batch_size, generator=None, bucket_multiplier=50):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if bucket_multiplier < 1:
            raise ValueError(
                f"bucket_multiplier must be positive, got {bucket_multiplier}")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.generator = generator
        self.bucket_multiplier = int(bucket_multiplier)

    def __len__(self):
        if not self.lengths:
            return 0
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        n = len(self.lengths)
        if n == 0:
            return
        generator = self.generator
        perm = torch.randperm(n, generator=generator).tolist()
        window = self.batch_size * self.bucket_multiplier
        batches = []
        for start in range(0, n, window):
            bucket = perm[start:start + window]
            bucket.sort(key=self.lengths.__getitem__)
            for offset in range(0, len(bucket), self.batch_size):
                batches.append(bucket[offset:offset + self.batch_size])
        order = torch.randperm(len(batches), generator=generator).tolist()
        for index in order:
            yield batches[index]
