import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedHLL(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            # Layer 4
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),

            # Output
            nn.Linear(64, 1)
        )

    def forward(self, M):
        return self.mlp(M).squeeze(1)
