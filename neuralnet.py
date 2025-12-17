import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedHLL(nn.Module):
    def __init__(self, input_dim=48):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, M):
        logN = self.mlp(x)
        return logN.squeeze(1)
