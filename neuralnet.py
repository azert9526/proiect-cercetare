import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedHLL(nn.Module):
    def __init__(self, max_reg=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_reg, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, max_reg)
        )

    def forward(self, hist):
        logits = self.net(hist)
        w = F.softmax(logits, dim=1)
        return w


