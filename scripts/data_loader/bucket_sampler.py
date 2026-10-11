import math
import torch
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        include_goal: bool = True,
        bucket_size_multiplier: int = 50,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.include_goal = include_goal
        self.bucket_size_multiplier = bucket_size_multiplier
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Estimate each sample's actual input length.
        self.lengths = []

        for i in range(len(dataset)):
            item = dataset[i]
            goal_len = len(item["goal"])
            sol1_len = len(item["sol1"])
            sol2_len = len(item["sol2"])

            if include_goal:
                len1 = goal_len + sol1_len + 3
                len2 = goal_len + sol2_len + 3
            else:
                len1 = sol1_len + 2
                len2 = sol2_len + 2

            self.lengths.append(max(len1, len2))

    def __iter__(self):
        n = len(self.dataset)

        if self.shuffle:
            indices = torch.randperm(n).tolist()
        else:
            indices = list(range(n))

        # Divide the shuffled indices into large local windows.
        bucket_size = self.batch_size * self.bucket_size_multiplier
        buckets = [
            indices[i:i + bucket_size]
            for i in range(0, n, bucket_size)
        ]

        batches = []

        for bucket in buckets:
            # Similar-length samples are placed next to each other.
            bucket.sort(key=lambda idx: self.lengths[idx])

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                batches.append(batch)

        # Randomize the order of batches during training.
        if self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in order]

        yield from batches

    def __len__(self):
        n = len(self.dataset)

        if self.drop_last:
            return n // self.batch_size

        return math.ceil(n / self.batch_size)