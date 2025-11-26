import os

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from generate_training_data import HLLDataset, generate_training_sample, generate_test_sample
from neuralnet import LearnedHLL
import matplotlib.pyplot as plt


def train_model(dataset_size=50000, batch_size=64, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnedHLL().to(device)

    if os.path.exists("learned_hll_weights.pth"):
        model.load_state_dict(torch.load("learned_hll_weights.pth", map_location=device))
        model.eval()
        print("model loaded")
        return model

    dataset = HLLDataset(dataset_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimiz = optim.Adam(model.parameters(), lr=1e-3)
    print(f"model built")

    for epoch in range(epochs):
        losses = []

        for batch_idx, (features, trueN) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"[DEBUG]  Batch {batch_idx}/{len(loader)}")

            features = features.to(device, non_blocking=True)
            trueN = trueN.to(device, non_blocking=True)

            optimiz.zero_grad()

            pred = model(features)
            pred = torch.clamp(pred, min=1.0)  # stability

            # Loss scaling (log MSE)
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
        if i % 5 == 0:
            print(f"Evaluate {i}/{num_eval}")
        features, hll_estimate, N = generate_test_sample()
        # baseline estimate from fresh HLL

        Ns.append(N)
        baseline_err.append(abs(hll_estimate - N) / N)

        device = next(model.parameters()).device
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(x).item()
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


