import torch
import torch.nn as nn
from torch.nn.functional import softplus

class LearnedHLL(nn.Module):
    def __init__(self, input_dim=73):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.relu(self.net(x))


