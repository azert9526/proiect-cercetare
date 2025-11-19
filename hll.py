import math
from hashlib import sha256
import numpy as np


def get_alpha(p):
    if not (4 <= p <= 16):
        raise ValueError("p=%d should be in range [4 : 16]" % p)

    if p == 4:
        return 0.673

    if p == 5:
        return 0.697

    if p == 6:
        return 0.709

    return 0.7213 / (1.0 + 1.079 / (1 << p))


def rho(w, p):
    leading_zeros = 64 - p - w.bit_length()

    if leading_zeros < 0:
        raise ValueError('w is too large')

    return leading_zeros + 1


def cardinality_estimation(S, p=16):
    """
    return HyperLogLog cardinality estimate for set of values with given precision
    :param S: set of values
    :param p: precision, between [4..16]
    """

    # Phase 0: Initialization
    alpha = get_alpha(p)
    p = p
    m = 1 << p # m = 2 ^ p
    M = np.zeros(m, np.int8)

    # Phase 1: Aggregation
    for value in S:
        x = int.from_bytes(sha256(value.encode('utf-8')).digest()[:8], byteorder='big')
        idx = x & (m - 1)
        w = x >> p
        M[idx] = max(M[idx], rho(w, p))

    # Phase 2: Result computation
    Z = 0
    for j in range(m):
        Z += 1 / (2 ** M[j])

    E = alpha * m * m / Z
    return E
