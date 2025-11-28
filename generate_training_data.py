import numpy as np
from hll import HyperLogLog, extract_features
import torch
from torch.utils.data import Dataset, DataLoader


def generate_training_sample(p=16, max_reg=64):
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

    # Histogram
    hist = np.bincount(M, minlength=max_reg).astype(np.float32)

    return M, hist, float(N)


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
