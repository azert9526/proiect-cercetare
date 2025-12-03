import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedHLL(nn.Module):
    def __init__(self, p=16, q=48):
        super().__init__()
        self.q = q

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=4),   # 65536 -> 16384
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=4),  # 16384 -> 4096
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=4),  # 4096 -> 1024
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(64 * 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, M):
        x = M.float() / self.q
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        logN = self.mlp(x)
        return logN.squeeze(1)

def extract_features(M, p=16, q=48):
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M)

    M = M.float()

    features = [
        M.mean(),
        M.std(),
        (M == 0).float().mean(),
        M.max(),
        M.median(),
        torch.quantile(M, 0.90),
        torch.quantile(M, 0.95),
        torch.quantile(M, 0.99)
    ]

    hist = torch.bincount(M.long(), minlength=q).float()
    features.extend(hist.tolist())

    return torch.tensor(features, dtype=torch.float32)
