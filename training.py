import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from generate_training_data import HLLPrecomputedDataset
from neuralnet import LearnedHLLModel
import matplotlib.pyplot as plt


def train_model(
    batch_size=1024,
    epochs=100,
    patience=15
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnedHLLModel().to(device)

    if os.path.exists("learned_hll_weights.pth"):
        model.load_state_dict(torch.load("learned_hll_weights.pth", map_location=device))
        model.eval()
        print("model loaded")
        return model

    train_ds = HLLPrecomputedDataset("data_train.npz")
    val_ds = HLLPrecomputedDataset("data_val.npz")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.SmoothL1Loss()

    pbar = trange(epochs, desc="Training")

    best_loss = float('inf')
    patience_counter = 0

    for epoch in pbar:
        model.train()
        train_loss = 0.0

        for hists, E_raw, targets in train_loader:
            hists, E_raw, targets = hists.to(device), E_raw.to(device), targets.to(device)

            optimizer.zero_grad()
            pred_correction = model(hists)

            loss = criterion(pred_correction, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hists, E_raw, targets in val_loader:
                hists, E_raw, targets = hists.to(device), E_raw.to(device), targets.to(device)
                pred = model(hists)
                val_loss += criterion(pred, targets).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        pbar.set_postfix({'Train': f'{avg_train:.4f}', 'Val': f'{avg_val:.6f}'})

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'best_enhanced_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), "learned_hll_weights.pth")
    print("model weights saved successfully")

    return model


