import numpy as np
import torch.nn.functional as F

from generate_training_data import HLLDataset, generate_training_sample, generate_dataset
from training import train_model, evaluate_and_plot
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "None")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


model = train_model()
evaluate_and_plot(model)

