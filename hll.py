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


class LearnedRegisterWeightedHLL(HyperLogLog):
    def __init__(self, model, p):
        super().__init__(p)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

    def estimate(self):
        Z = np.sum(1.0 / (2.0 ** self.M))
        hist = np.bincount(self.M, minlength=64).astype(np.float32)
        hist = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_corr = self.model(hist).squeeze(1)     # shape (1,)
            corr = raw_corr[0].item()

        E = corr * self.m * self.m / Z
        return E


def extract_features(M, max_reg=64):
    m = len(M)
    hist = np.bincount(M, minlength=max_reg)

    features = [
        M.mean(),
        M.std(),
        np.sum(M == 0) / m,
        np.sum(M == 1) / m,
        np.max(M),
        np.median(M),
        np.percentile(M, 90),
        np.percentile(M, 95),
        np.percentile(M, 99)
    ]

    features.extend(hist.tolist())
    return np.array(features, dtype=np.float32)
