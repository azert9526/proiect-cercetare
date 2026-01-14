import cupy as cp
import numpy as np
import time

# --- 1. The Ultra-Fast Kernel (Inline RNG) ---
hll_fast_kernel = cp.RawKernel(r'''
extern "C" __global__
void hll_generate(
    unsigned int* registers,      // output: HLL registers
    unsigned long long seed,      
    unsigned long long N,         // items to simulate
    int p                         // precision
) {
    unsigned long long tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    //rng
    unsigned long long state = seed + tid;
    unsigned long long z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    state = z ^ (z >> 31);

    for (unsigned long long i = tid; i < N; i += stride) {

        // xorshift64
        unsigned long long x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        unsigned long long h = x * 0x2545F4914F6CDD1D; // random hash

        // hll
        unsigned long long idx = h >> (64 - p);
        unsigned long long w = h << p;

        // clz
        int lz = (w == 0) ? 64 : __clzll(w);
        unsigned int rho = lz + 1;

        atomicMax(&registers[idx], rho);
    }
}
''', 'hll_generate')


def generate_single_hll_fast(p=16, batch_size=None):
    log_N = np.random.uniform(0, 10)
    N = int(10 ** log_N)

    m = 1 << p
    M = cp.zeros(m, dtype=cp.uint32)

    threads = 512
    blocks = 2048

    seed = np.random.randint(0, 2 ** 63)

    hll_fast_kernel(
        (blocks,), (threads,),
        (M, int(seed), int(N), int(p))
    )

    return cp.asnumpy(M).astype(np.uint8), float(N)