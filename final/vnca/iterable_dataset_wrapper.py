import random

from torch.utils.data import IterableDataset, Dataset

# -------- Taken from https://github.com/rasmusbergpalm/vnca/blob/main/modules/iterable_dataset_wrapper.py

class IterableWrapper(IterableDataset):
    """
    Turns a Dataset into an IterableDataset by endlessly yielding random samples from the dataset.
    """

    def __init__(self, delegate: Dataset):
        self.delegate = delegate

    def __iter__(self):
        l = len(self.delegate)
        while True:
            for idx in random.sample(range(l), l):
                yield self.delegate[idx]
