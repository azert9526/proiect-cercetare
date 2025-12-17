import numpy as np
import cupy as cp
import torch.nn.functional as F
from torch.utils.data import random_split
import torch
from tqdm import tqdm

from generate_training_data import HLLDataset, generate_training_sample, generate_dataset, get_test_loader
from training import train_model, evaluate_and_plot

#p - precision (number of registers)
#q - value range of register (total_bits - p)

model = train_model()
test_loader = get_test_loader()
evaluate_and_plot(model, test_loader)

