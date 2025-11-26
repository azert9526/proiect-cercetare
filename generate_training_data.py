import numpy as np
from hll import HyperLogLog, extract_features
import torch
from torch.utils.data import Dataset, DataLoader


def generate_training_sample(p=16, max_reg=64):
    # Sample N ~ log-uniform
    N = int(10 ** np.random.uniform(1, 8))

    m = 1 << p
    lam = N / m

    # 1) Draw Poisson counts for each register
    K = np.random.poisson(lam=lam, size=m)

    # 2) Registers with K=0 stay at 0
    M = np.zeros(m, dtype=np.uint8)

    # 3) For K>0, register = 1 + Geometric(p=0.5)
    positive = K > 0
    # numpy geometric gives number of trials until first success
    M[positive] = 1 + np.random.geometric(p=0.5, size=positive.sum())

    # 4) Extract your features
    features = extract_features(M, max_reg=max_reg)

    return features, float(N)


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

    # --- Compute the true HLL estimate ---
    hll_est = estimate_hll(M, p)

    # --- Extract your features ---
    features = extract_features(M, max_reg=max_reg)

    return features, hll_est, float(N)


def estimate_hll(M, p):
    m = 1 << p
    alpha_m = 0.7213/(1+1.079/m)
    Z = np.sum(2.0 ** (-M))
    E = alpha_m * m*m / Z

    # small range correction
    if E <= 2.5*m:
        V = np.sum(M == 0)
        if V > 0:
            E = m * np.log(m / V)

    # large range correction
    if E > (1/30) * (1 << 32):
        E = - (1 << 32) * np.log(1 - E / (1 << 32))

    return E


class HLLDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        features, N = generate_training_sample()
        return torch.tensor(features), torch.tensor([N], dtype=torch.float32)
