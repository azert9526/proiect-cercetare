import numpy as np
import torch
import torch.nn as nn


class LearnedHLLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(49, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),

            # Output: Log10 Correction Factor
            nn.Linear(64, 1)
        )

    def forward(self, hist):
        occupancy = 1.0 - hist[:, 0:1]
        log_occupancy = torch.log10(occupancy + 1e-6)

        x = torch.cat([hist, log_occupancy], dim=1)

        return self.mlp(x).squeeze(1)
