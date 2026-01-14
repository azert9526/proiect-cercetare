import numpy as np
import cupy as cp
import torch.nn.functional as F
from torch.utils.data import random_split
import torch
from tqdm import tqdm

from generate_training_data import get_test_loader
from training import train_model
from evaluate import eval_all_hll, run_shakespeare_benchmark

#p - precision (number of registers)
#q - value range of register (total_bits - p)

model = train_model()
run_shakespeare_benchmark(model)
#test_loader = get_test_loader()
#eval_all_hll(model, test_loader)

