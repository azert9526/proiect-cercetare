import numpy as np
import torch.nn.functional as F

from generate_training_data import HLLDataset, generate_training_sample, generate_dataset, count_leading_zeros
from training import train_model, evaluate_and_plot
import torch

print("Torch device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#p - precision (number of registers)
#q - value range of register (total_bits - p)

generate_dataset()
#model = train_model()
#evaluate_and_plot(model)

