import math
from hashlib import sha256
import numpy as np


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
        return E

    def __len__(self):
        return int(self.estimate())


class LearnedRegisterWeightedHLL(HyperLogLog):
    def __init__(self, model, p):
        super().__init__(p)
        self.model = model

    def get_features(self):
        M = self.M

        hist = np.bincount(M, minlength=M.max(initial=0) + 1) / self.m

        features = [
            M.mean(),
            M.std(),
            np.sum(M == 0) / self.m,
            np.sum(M == 1) / self.m,
            np.max(M),
            np.median(M),
        ]
        features.extend(hist.tolist())

        return np.array(features, dtype=float).reshape(1, -1)

    def weighted_estimate(self):
        """
        Compute estimate using learned register weights.
        """
        max_register_value = int(self.M.max(initial=0))
        bucket_sums = np.zeros(max_register_value + 1, float)
        for v in range(max_register_value + 1):
            bucket_sums[v] = np.sum(self.M == v)

        features = self.get_features()
        raw_w = self.model.predict(features)
        raw_w = np.asarray(raw_w).reshape(-1)

        w = np.clip(raw_w, a_min=0.0, a_max=None)
        w = w / np.sum(w)

        if len(raw_w) != bucket_sums:
            raise ValueError(
                f"length of {len(raw_w)} weights =/= number of registers m = {self.m}"
            )

        Z = np.sum(w * (2.0 ** (-bucket_sums)))

        E = self.alpha * self.m * self.m / Z

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
