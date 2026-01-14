import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def compute_raw_hll(histograms, p=16):
    m = 1 << p
    # constants for p=16
    alpha = 0.7213 / (1 + 1.079 / m)
    const = alpha * (m ** 2)

    powers = 2.0 ** (-np.arange(64))
    Z_inv = np.dot(histograms, powers)  # shape: (batch_size,)

    E = const / (Z_inv * m)  # formula adjustment for normalized hist (*m)
    return E

class HLLPrecomputedDataset(Dataset):
    def __init__(self, path="data_train.npz", p=16):
        data = np.load(path)
        self.histograms = torch.from_numpy(data['histograms']).float() #these are normalized
        self.histograms = self.histograms[:, :48] #cutting last p=16 values

        true_N = data['cardinalities']
        self.E_raw = torch.from_numpy(compute_raw_hll(data['histograms'], p=p)).float()

        # target is correction = true_N / E
        # log(correction) = log(true_N / E)
        self.targets = torch.from_numpy(np.log10(true_N / self.E_raw.numpy())).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.histograms[idx], self.E_raw[idx], self.targets[idx]


def get_test_loader(batch_size=1024):
    data = HLLPrecomputedDataset("data_test.npz")

    return DataLoader(data, batch_size=batch_size, shuffle=False)
