import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from generate_training_data import HLLDataset, generate_training_sample, generate_test_sample, HLLPrecomputedDataset
from hll import HyperLogLog, LearnedRegisterWeightedHLL
from neuralnet import LearnedHLL, extract_features
import matplotlib.pyplot as plt


def train_model(
    batch_size=64,
    epochs=3,
    p=16,
    q=48
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnedHLL().to(device)

    if os.path.exists("learned_hll_weights.pth"):
        model.load_state_dict(torch.load("learned_hll_weights.pth", map_location=device))
        model.eval()
        print("model loaded")
        return model

    dataset = HLLPrecomputedDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimiz = optim.Adam(model.parameters())
    print(f"model built")

    progress_epoch = tqdm(range(epochs), desc="Total time completed")
    for epoch in progress_epoch:
        losses = []

        # tqdm progress bar
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)

        for M_batch, trueN in progress:
            M_batch = M_batch.to(device)
            trueN = trueN.to(device, non_blocking=True)

            # forward + backward
            optimiz.zero_grad()

            log_pred = model(M_batch)
            log_true = torch.log(trueN)

            loss = mse_loss(log_pred, log_true)
            loss.backward()
            optimiz.step()

            losses.append(loss.item())

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
            })

        progress_epoch.set_postfix({
            f"mean loss": f"{np.mean(losses):.6f}"
        })

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

def evaluate_and_plot(model, p=16, num_eval=400):
    Ns = []
    baseline_err = []
    learned_err = []

    for _ in tqdm(range(num_eval)):
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


