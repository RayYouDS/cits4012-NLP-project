"""Length-bucketed batch sampler for the training set.

Each PIQA example contains approximately 35 tokens. If a batch contains a
single long sequence, all sequences in that batch must be padded to its length.
On the training set split by goal, the average padded length with randomly
shuffled batches of size 64 was 135.6. Sorting examples by length within a
window before forming batches reduced the average padded length to 40.1.
Do not use this sampler for validation or test sets.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Sampler


def piqa_lengths(df):
    """Calculate the padded length of each example: goal + longer solution + [CLS]/[SEP]/[EOS]."""
    goal = df["goal_tokens"].map(len).to_numpy()
    sol1 = df["sol1_tokens"].map(len).to_numpy()
    sol2 = df["sol2_tokens"].map(len).to_numpy()
    return (goal + np.maximum(sol1, sol2) + 3).astype(int).tolist()


class LengthBucketBatchSampler(Sampler):
    """For each epoch, shuffle indices, sort by length within each window, form batches, then shuffle batch order."""

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
