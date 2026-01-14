import math
from hashlib import sha256
import numpy as np
import torch


class HyperLogLog:
    def __init__(self, p):
        if not (4 <= p <= 16):
            raise ValueError(f"p={p} should be in range [4 : 16]")

        self.p = p
        self.m = 1 << p
        self.alpha = self._get_alpha(p)
        self.M = np.zeros(self.m, np.int8)

    @staticmethod
    def _get_alpha(p):
        if p == 4:
            return 0.673
        if p == 5:
            return 0.697
        if p == 6:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / (1 << p))

    @staticmethod
    def _rho(w, p):
        leading_zeros = 64 - p - w.bit_length()
        if leading_zeros < 0:
            raise ValueError("w is too large")
        return leading_zeros + 1

    def add(self, value: str):
        """Add a single value (string) to the sketch."""
        x = int.from_bytes(sha256(value.encode("utf-8")).digest()[:8], byteorder="big")
        idx = x & (self.m - 1)
        w = x >> self.p
        self.M[idx] = max(self.M[idx], self._rho(w, self.p))

    def add_many(self, iterable):
        for v in iterable:
            self.add(v)

    def estimate(self):
        Z = np.sum(1.0 / (2.0 ** self.M))
        E = self.alpha * self.m * self.m / Z

        # small range correction
        if E <= 2.5 * self.m:
            V = np.sum(self.M == 0)
            if V > 0:
                E = self.m * np.log(self.m / V) # linear counting

        # large range correction
        if E > (1 / 30) * (1 << 32):
            E = - (1 << 32) * np.log(1 - E / (1 << 32))

        return E

    def set_registers(self, M, p):
        self.M = M
        self.p = p

    def __len__(self):
        return int(self.estimate())


class LearnedHLL(HyperLogLog):
    def __init__(self, model, p):
        super().__init__(p)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.eval()

    def estimate(self):
        Z_inv = np.sum(2.0 ** (-self.M))
        E = self.alpha * (self.m * self.m) / Z_inv

        hist = np.bincount(self.M, minlength=48)[:48].astype(np.float32) #histogram of registers
        hist = hist / self.m  # normalize

        hist_tensor = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(self.device)

        # inference
        with torch.no_grad():
            pred_log_delta = self.model(hist_tensor).item()

        estimated_cardinality = E * (10 ** pred_log_delta)

        return estimated_cardinality


def get_hll_constants(p):
    m = 1 << p
    if p == 4: alpha = 0.673
    elif p == 5: alpha = 0.697
    elif p == 6: alpha = 0.709
    else: alpha = 0.7213 / (1 + 1.079 / m)
    return alpha * m * m, m


def hll_estimate_from_histograms(histograms, p=16):
    const, m = get_hll_constants(p)

    # rescale normalized histograms
    counts = histograms * m

    exponents = np.arange(64)
    powers = 2.0 ** (-exponents)

    Z_inv = np.dot(counts, powers)

    E = const / Z_inv

    final_estimates = np.zeros_like(E)

    # linear counting
    V = counts[:, 0]

    # mask for small range (E < 2.5m) and empty registers exist (V > 0)
    small_mask = (E < 2.5 * m) & (V > 0)
    final_estimates[small_mask] = m * np.log(m / V[small_mask])

    # large range
    final_estimates[~small_mask] = E[~small_mask]

    return final_estimates
