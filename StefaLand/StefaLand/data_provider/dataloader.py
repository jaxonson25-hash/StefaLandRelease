# from torch.utils.data import DataLoader
import numpy as np

class DataLoader:

    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

    def __getitem__(self, i):

        if i >= len(self.dataset):
            raise IndexError("Index out of range")

        batch = self.dataset[i]

        return batch

    def __len__(self):
        return len(self.indexes)