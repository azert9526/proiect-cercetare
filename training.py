import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm

from generate_training_data import HLLDataset, generate_training_sample, generate_test_sample, HLLPrecomputedDataset
from hll import HyperLogLog, LearnedRegisterWeightedHLL, hll_estimate_from_histograms
from neuralnet import LearnedHLL
import matplotlib.pyplot as plt


def train_model(
    batch_size=1024,
    epochs=50
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnedHLL().to(device)

    if os.path.exists("learned_hll_weights.pth"):
        model.load_state_dict(torch.load("learned_hll_weights.pth", map_location=device))
        model.eval()
        print("model loaded")
        return model

    train_data = np.load("data_train.npz")
    val_data = np.load("data_val.npz")

    train_ds = TensorDataset(torch.from_numpy(train_data['histograms']).float(),
                             torch.from_numpy(np.log10(train_data['cardinalities'])).float())
    val_ds = TensorDataset(torch.from_numpy(val_data['histograms']).float(),
                           torch.from_numpy(np.log10(val_data['cardinalities'])).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.L1Loss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Train
        for hists, targets in train_loader:
            hists, targets = hists.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(hists)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hists, targets in val_loader:
                hists, targets = hists.to(device), targets.to(device)
                preds = model(hists)
                val_loss += criterion(preds, targets).item()

        avg_val = val_loss / len(val_loader)

        # Update Scheduler
        scheduler.step(avg_val)

        # Save Best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_hll_model.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    print("Done. Loading best weights...")
    model.load_state_dict(torch.load("best_hll_model.pth"))
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

def evaluate_and_plot(model, test_loader, p=16):
    device = next(model.parameters()).device
    model.eval()

    true_Ns = []
    pred_Ns_learned = []
    pred_Ns_baseline = []

    print("Running evaluation on Test Set...")
    with torch.no_grad():
        for histograms, log_targets in tqdm(test_loader):
            histograms_gpu = histograms.to(device)
            log_preds = model(histograms_gpu).cpu().numpy()
            learned_est = 10 ** log_preds

            hists_np = histograms.numpy()
            baseline_est = hll_estimate_from_histograms(hists_np, p=p)

            true_N = 10 ** log_targets.numpy()

            true_Ns.extend(true_N)
            pred_Ns_learned.extend(learned_est)
            pred_Ns_baseline.extend(baseline_est)

    true_Ns = np.array(true_Ns)
    pred_learned = np.array(pred_Ns_learned)
    pred_baseline = np.array(pred_Ns_baseline)

    err_learned = np.abs(pred_learned - true_Ns) / true_Ns
    err_baseline = np.abs(pred_baseline - true_Ns) / true_Ns

    print(f"Baseline MAPE: {np.mean(err_baseline) * 100:.4f}%")
    print(f"Learned MAPE:  {np.mean(err_learned) * 100:.4f}%")

    # --- PLOTTING ---
    plot_results(true_Ns, err_baseline, err_learned)


def plot_results(true_Ns, err_baseline, err_learned):
    plt.figure(figsize=(12, 6))

    # Bin the data to show trends clearly (Scatter plots hide density)
    # We create log-spaced bins
    bins = np.logspace(np.log10(min(true_Ns)), np.log10(max(true_Ns)), 50)

    # Calculate mean error in each bin
    bin_centers = []
    mean_err_base = []
    mean_err_learn = []

    for i in range(len(bins) - 1):
        mask = (true_Ns >= bins[i]) & (true_Ns < bins[i + 1])
        if np.any(mask):
            bin_centers.append(np.sqrt(bins[i] * bins[i + 1]))  # Geometric mean center
            mean_err_base.append(np.mean(err_baseline[mask]))
            mean_err_learn.append(np.mean(err_learned[mask]))

    # Plot 1: Scatter (High transparency)
    plt.subplot(1, 2, 1)
    plt.scatter(true_Ns, err_baseline, alpha=0.05, s=1, color='blue', label='Baseline (Raw)')
    plt.scatter(true_Ns, err_learned, alpha=0.05, s=1, color='red', label='Learned (Raw)')

    # Plot Trend Lines (The important part)
    plt.plot(bin_centers, mean_err_base, color='blue', linewidth=2, label='Baseline Mean')
    plt.plot(bin_centers, mean_err_learn, color='red', linewidth=2, label='Learned Mean')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cardinality (N)')
    plt.ylabel('Relative Error')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Boxplot or Smoothed Comparison
    plt.subplot(1, 2, 2)
    plt.plot(bin_centers, mean_err_base, color='blue', linestyle='--', label='Baseline')
    plt.plot(bin_centers, mean_err_learn, color='red', label='Learned')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cardinality (N)')
    plt.ylabel('Mean Relative Error')
    plt.title('Trend Comparison (Smoothed)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()


