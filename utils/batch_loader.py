import numpy as np
import torch
from torch.utils.data import Dataset


class BatchLoader(object):

    def __init__(self, dataset: Dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = int(batch_size)
        self.reset_index()

    def reset_index(self):
        num_samples = len(self.dataset)
        if self.shuffle:
            index = torch.randperm(num_samples)
        else:
            index = torch.arange(num_samples)
        self.index = torch.chunk(index.to(torch.long), int(np.ceil(num_samples / self.batch_size)))
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= len(self.index):
            raise StopIteration
        batch = self.dataset[self.index[self.current_batch]]
        self.current_batch += 1
        return batch

    def __iter__(self):
        return self.reset_index()

    def __len__(self):
        return len(self.index)
