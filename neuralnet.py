import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedHLL(nn.Module):
    def __init__(self, max_reg=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_reg, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, hist):
        hist = torch.log1p(hist)
        hist = hist / hist.sum(dim=1, keepdim=True)

        raw_corr = self.net(hist)
        corr = 1.0 + 0.5 * torch.tanh(raw_corr)
        return corr


