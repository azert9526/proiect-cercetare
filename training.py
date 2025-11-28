import os

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from generate_training_data import HLLDataset, generate_training_sample, generate_test_sample
from hll import HyperLogLog, extract_features, LearnedRegisterWeightedHLL
from neuralnet import LearnedHLL
import matplotlib.pyplot as plt


def train_model(
    dataset_size=50000,
    batch_size=64,
    epochs=5,
    p=16,
    max_reg=64,
    lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnedHLL(max_reg=max_reg).to(device)

    if os.path.exists("learned_hll_weights.pth"):
        model.load_state_dict(torch.load("learned_hll_weights.pth", map_location=device))
        model.eval()
        print("model loaded")
        return model

    dataset = HLLDataset(dataset_size, p=p, max_reg=max_reg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimiz = optim.Adam(model.parameters(), lr=lr)
    print(f"model built")

    m = 1 << p
    alpha = 0.7213 / (1.0 + 1.079 / m)

    v = torch.arange(max_reg, device=device, dtype=torch.float32)
    pow_term = torch.pow(2.0, -v)

    for epoch in range(epochs):
        losses = []

        for batch_idx, (hist, M, trueN) in enumerate(loader):

            hist = hist.to(device, non_blocking=True)  # (B, max_reg)
            trueN = trueN.to(device, non_blocking=True)  # (B,)

            optimiz.zero_grad()

            # 1) predict weights
            w = model(hist)  # (B, max_reg)

            # 2) compute Z = sum_v w_v * count_v * 2^{-v}
            # hist: counts per v
            Z = torch.sum(w * hist * pow_term, dim=1)  # (B,)
            Z = torch.clamp(Z, min=1e-12)

            # 3) HLL-like estimate
            pred = alpha * m * m / Z  # (B,)

            if batch_idx % 50 == 0:
                print(f"[DEBUG]  Batch {batch_idx}/{len(loader)}")
                print("pred =", pred[0].item(), "trueN =", trueN[0].item())

            # 4) log-MSE loss
            loss = mse_loss(torch.log1p(pred), torch.log1p(trueN))

            loss.backward()
            optimiz.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.6f}")

    torch.save(model.state_dict(), "learned_hll_weights.pth")
    print("Model saved!")
    return model


def eval_model(model, samples=5000):
    errs = []

    for _ in range(samples):
        features, N = generate_training_sample()
        x = torch.tensor(features).unsqueeze(0)
        pred = model(x).item()
        errs.append(abs(pred - N) / N)

    print("Mean Relative Error:", np.mean(errs))
    print("Median Relative Error:", np.median(errs))

def evaluate_and_plot(model, p=16, num_eval=200):
    Ns = []
    baseline_err = []
    learned_err = []

    for i in range(num_eval):
        if i % 10 == 0:
            print(f"Evaluate {i}/{num_eval}")
        M, N = generate_test_sample()
        # baseline estimate from fresh HLL

        hll = HyperLogLog(p)
        hll.set_registers(M, p)
        hll_estimate = hll.estimate()

        Ns.append(N)
        baseline_err.append(abs(hll_estimate - N) / N)

        learned_hll = LearnedRegisterWeightedHLL(model, p)
        learned_hll.set_registers(M, p)
        pred = learned_hll.estimate()
        learned_err.append(abs(pred - N) / N)

    Ns = np.array(Ns)
    baseline_err = np.array(baseline_err)
    learned_err = np.array(learned_err)

    print("Baseline mean rel error:", baseline_err.mean(), "Learned mean rel error:", learned_err.mean())
    # plot error vs N
    plt.figure(figsize=(7, 4))
    plt.scatter(Ns, baseline_err, label='baseline', alpha=0.6, s=20)
    plt.scatter(Ns, learned_err, label='learned', alpha=0.6, s=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("True N")
    plt.ylabel("Relative error")
    plt.legend()
    plt.title("Baseline vs Learned relative error scatter")
    plt.show()


