import numpy as np
from hll import HyperLogLog, extract_features
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def generate_training_sample(p=16, max_reg=64):
    N = int(10 ** np.random.uniform(1, 8))
    m = 1 << p

    # Poisson number of items per register
    lam = N / m
    X = np.random.poisson(lam, size=m)

    # Vectorized sampling of max geometric per register
    U = np.random.rand(m)
    # Avoid div-by-zero for registers with X=0
    X_safe = np.maximum(X, 1)

    M = np.log(1 - U**(1.0 / X_safe)) / np.log(0.5)
    M[X == 0] = 0     # fixed value
    M = np.clip(M, 0, max_reg - 1).astype(np.int32)

    # Histogram
    hist = np.bincount(M, minlength=max_reg).astype(np.float32)

    return M, hist, float(N)


def generate_dataset(num_samples=100000, p=16, max_reg=64):
    hists = np.zeros((num_samples, max_reg), dtype=np.float32)
    cardinalities = np.zeros(num_samples, dtype=np.float32)

    print("Generating dataset...")
    for i in tqdm(range(num_samples)):
        _, hist, N = generate_training_sample(p=p, max_reg=max_reg)
        hists[i] = hist
        cardinalities[i] = N

    print("Saving dataset to training_data.npz...")
    np.savez_compressed("training_data.npz", hists=hists, cardinalities=cardinalities)
    print("Done! File: training_data.npz")


# Precompute leading zeros for all byte values 0..255
LZ_TABLE = np.array([8] + [bin(i).find('1') for i in range(1, 256)], dtype=np.uint8)

def clz64_numpy(x):
    """
    Count leading zeros for an array of uint64 using vectorized operations.
    Returns array of same shape.
    """

    # Convert to bytes: shape (..., 8)
    b = x.view(np.uint8).reshape(-1, 8)

    # Find index of first non-zero byte
    nz = b != 0
    has_val = nz.any(axis=1)

    # For entries with no nonzero bytes (x = 0)
    # Return 64 leading zeros
    result = np.zeros(len(x), dtype=np.uint8)
    result[~has_val] = 64

    # For nonzero bytes
    # Get index of first non-zero byte
    first_byte = np.argmax(nz, axis=1)

    # Count leading zeros inside that byte
    first_byte_values = b[np.arange(len(x)), first_byte]
    lz_inside = LZ_TABLE[first_byte_values]

    # Total leading zeros = byte_index*8 + lz_inside
    result[has_val] = first_byte[has_val] * 8 + lz_inside[has_val]
    return result

def generate_test_sample(p=16, max_reg=64):
    N = int(10 ** np.random.uniform(1, 8))
    m = 1 << p

    # --- Generate random 64-bit hashes ---
    h = np.random.randint(0, (1 << 63), size=N, dtype=np.uint64)

    # --- Split hash into index + remainder bits ---
    idx = h >> (64 - p)
    w = (h << p) & np.uint64((1 << 64) - 1)

    # --- Vectorized leading zero count ---
    lz = clz64_numpy(w)
    rho = lz + 1     # HLL uses 1 + leading zeros

    # --- Build registers ---
    M = np.zeros(m, dtype=np.uint8)
    np.maximum.at(M, idx, rho.astype(np.uint8))

    return M, float(N)


class HLLDataset(Dataset):
    def __init__(self, size, p=16, max_reg=64):
        self.size = size
        self.p = p
        self.max_reg = max_reg

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        M, hist, N = generate_training_sample(p=self.p, max_reg=self.max_reg)

        hist = torch.tensor(hist, dtype=torch.float32)     # (max_reg,)
        M = torch.tensor(M, dtype=torch.long)              # (m,)
        N = torch.tensor(N, dtype=torch.float32)           # scalar

        return hist, M, N


class HLLPrecomputedDataset(Dataset):
    def __init__(self, path="training_data.npz"):
        data = np.load(path)
        self.hists = data["hists"]
        self.trueN = data["cardinalities"]

    def __len__(self):
        return len(self.hists)

    def __getitem__(self, idx):
        hist = torch.from_numpy(self.hists[idx])
        N = torch.tensor(self.trueN[idx], dtype=torch.float32)
        return hist, N
