import os
from hashlib import sha256

import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

import requests
import re

# Patch numpy int if needed
if not hasattr(np, 'int'):
    np.int = int
from python_hll.hll import HLL
from my_hll import LearnedHLL, HyperLogLog


def compute_raw_hll_numpy(histograms, p=16):
    """
    Computes standard HLL estimates from a batch of histograms
    """
    m = 1 << p
    alpha = 0.7213 / (1 + 1.079 / m)
    const = alpha * (m ** 2)

    powers = 2.0 ** (-np.arange(64))

    Z_inv_norm = np.dot(histograms, powers)
    E = const / (Z_inv_norm * m)

    return E


def eval_all_hll(model, test_loader, p=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    true_Ns = []
    err_learned = []
    err_linear_counting = []

    m = 1 << p

    print("Running Model & Linear Counting Predictions...")

    with torch.no_grad():
        for hists, E_raw, targets in tqdm(test_loader, desc="Inference"):
            hists = hists.to(device)
            E_raw = E_raw.to(device)
            targets = targets.numpy()

            pred_log = model(hists).cpu().numpy()

            E_np = E_raw.cpu().numpy()
            pred_N = E_np * (10 ** pred_log)
            batch_true_N = E_np * (10 ** targets)

            true_Ns.extend(batch_true_N)
            err_learned.extend(np.abs(pred_N - batch_true_N) / batch_true_N)

            # V = count of empty registers (index 0)
            # hists are normalized, multiply by m to get counts
            V_counts = hists[:, 0].cpu().numpy() * m

            E_lc = np.zeros_like(E_np)

            mask_lc = V_counts > 0

            E_lc[mask_lc] = m * np.log(m / V_counts[mask_lc])

            E_lc[~mask_lc] = E_np[~mask_lc]

            err_linear_counting.extend(np.abs(E_lc - batch_true_N) / batch_true_N)

    true_Ns = np.array(true_Ns)
    err_learned = np.array(err_learned)
    err_linear_counting = np.array(err_linear_counting)

    safe_indices = np.where(true_Ns <= 1_000_000)[0]

    num_samples = min(1500, len(safe_indices))
    indices = np.random.choice(safe_indices, num_samples, replace=False)

    subset_Ns = true_Ns[indices]
    err_hllpp = []

    cache_file = "hllpp_benchmark.npz"
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        subset_Ns = data['subset_Ns']
        err_hllpp = data['err_hllpp']
        print(f"   Loaded {len(subset_Ns)} benchmark points.")
    else:

        for N in tqdm(subset_Ns):
            N = int(N)
            if N == 0:
                err_hllpp.append(0)
                continue

            hllpp = HLL(p, 6)
            # Generate full range integers
            raw_ints = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=N, dtype=np.int64)
            for val in raw_ints:
                hllpp.add_raw(int(val))

            est = hllpp.cardinality()
            err_hllpp.append(abs(est - N) / N)

        err_hllpp = np.array(err_hllpp)
        print(f"   Saving results to '{cache_file}'...")
        np.savez(cache_file, subset_Ns=subset_Ns, err_hllpp=err_hllpp)



    plt.figure(figsize=(12, 7))

    def get_smooth_curve(Ns, Errs, window_size):
        idx = np.argsort(Ns)
        sorted_N = Ns[idx]
        sorted_Err = Errs[idx]
        smooth_Err = np.convolve(sorted_Err, np.ones(window_size) / window_size, mode='valid')
        # Adjust N to match size
        valid_N = sorted_N[window_size - 1:]
        return valid_N, smooth_Err

    lc_N, lc_Err = get_smooth_curve(true_Ns, err_linear_counting, window_size=1000)
    plt.plot(lc_N, lc_Err, color='green', linestyle='--', linewidth=2, label='Original HLL')

    hll_N, hll_Err = get_smooth_curve(subset_Ns, err_hllpp, window_size=100)
    plt.plot(hll_N, hll_Err, color='blue', linewidth=2, label='HLL++')

    learn_N, learn_Err = get_smooth_curve(true_Ns, err_learned, window_size=1000)
    plt.plot(learn_N, learn_Err, color='red', linewidth=2, label='Learned Model')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cardinality (N)')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.ylim(1e-4, 0.1)

    plt.xlim(100, 10_000_000)

    plt.show()


def run_shakespeare_benchmark(model, p=16):
    print(f"\n--- Running Shakespeare Benchmark (p={p}) ---")

    url = "https://www.gutenberg.org/files/100/100-0.txt"  # Complete Works of Shakespeare
    print("Downloading text from Project Gutenberg...")
    try:
        text = requests.get(url).text
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # tokenize
    # convert to lowercase for case-insensitivity
    words = re.findall(r'\w+', text.lower())
    print(f"Total words found: {len(words)}")

    # ground truth
    true_set = set(words)
    true_N = len(true_set)
    print(f"True Distinct Words (N): {true_N}")

    learned_hll = LearnedHLL(model, p)

    print("Adding words to Learned HLL Sketch...")
    learned_hll.add_many(words)

    est_N = learned_hll.estimate()

    rel_error = abs(est_N - true_N) / true_N
    print("-" * 30)
    print(f"True Cardinality:      {true_N}")
    print(f"Learned Estimate:      {int(est_N)}")
    print(f"Absolute Learned Error:{int(abs(est_N - true_N))}")
    print(f"Relative Learned Error:{rel_error * 100:.4f}%")

    hll = HyperLogLog(p=16)
    hll.add_many(words)

    est_hll = hll.estimate()
    rel_error = abs(est_hll - true_N) / true_N
    print(f"HLL Estimate:          {est_hll}")
    print(f"Absolute HLL Error:    {int(abs(est_hll - true_N))}")
    print(f"Relative HLL Error:    {rel_error * 100:.4f}%")
    print("-" * 30)
