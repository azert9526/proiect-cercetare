import numpy as np
import cupy as cp
from hll import HyperLogLog
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def count_leading_zeros(x, q):
    result = cp.zeros_like(x, dtype=cp.uint64)
    not_zero_mask = x != 0
    result[not_zero_mask] = q - (cp.floor(cp.log2(x[not_zero_mask])).astype(cp.uint64) + 1)
    result[~not_zero_mask] = q

    return result


"""def generate_training_sample(p=16, q=48):
    N = int(10 ** np.random.uniform(1, 8))
    m = 1 << p

    h = cp.random.randint(0, (1 << 63), size=N, dtype=cp.uint64)

    idx = h >> (64 - p)
    w = (h << p) & cp.uint64((1 << 64) - 1)

    lz = count_leading_zeros(w, q)
    rho = (lz + 1).astype(dtype=cp.uint64)

    M = cp.zeros(m, dtype=cp.uint64)
    cp.maximum.at(M, idx, rho)

    return cp.asnumpy(M).astype(dtype=np.uint8), float(N)"""

def generate_training_sample(p=16, q=48):
    N = int(10 ** np.random.uniform(1, 8))
    m = 1 << p

    lam = N / m

    # Poisson on GPU
    X = cp.random.poisson(lam, size=m)

    # Uniforms for max-geometric computation
    U = cp.random.random(m)

    X_safe = cp.maximum(X, 1)

    M = cp.log(1 - U**(1.0 / X_safe)) / cp.log(0.5)
    M = cp.where(X == 0, 0, M)

    # Final clipping
    M = cp.clip(M, 0, q).astype(cp.uint8)

    return cp.asnumpy(M), float(N)

def generate_dataset(num_samples=200000, p=16, q=48):
    m = 1 << p
    registers = np.zeros((num_samples, m), dtype=np.uint8)
    cardinalities = np.zeros(num_samples, dtype=np.uint64)

    print("Generating dataset...")
    for i in tqdm(range(num_samples)):
        M, N = generate_training_sample(p, q)
        registers[i] = M
        cardinalities[i] = N

    print("Saving dataset to training_data.npz...")
    np.savez_compressed("training_data.npz", registers=registers, cardinalities=cardinalities)
    print("Done! File: training_data.npz")


def generate_test_sample(p=16, q=48):
    return generate_training_sample(p, q)


class HLLDataset(Dataset):
    def __init__(self, size, p=16, max_reg=64):
        self.size = size
        self.p = p
        self.max_reg = max_reg

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        M, hist, N = generate_training_sample(p=self.p, max_reg=self.max_reg)

        hist = torch.tensor(hist, dtype=torch.float32)  # (max_reg,)
        M = torch.tensor(M, dtype=torch.long)  # (m,)
        N = torch.tensor(N, dtype=torch.float32)  # scalar

        return hist, M, N


class HLLPrecomputedDataset(Dataset):
    def __init__(self, path="training_data.npz"):
        data = np.load(path)
        self.registers = data["registers"]
        self.cardinalities = data["cardinalities"]

    def __len__(self):
        return len(self.cardinalities)

    def __getitem__(self, idx):
        M = torch.from_numpy(self.registers[idx])
        N = torch.tensor(self.cardinalities[idx], dtype=torch.float32)
        return M, N
