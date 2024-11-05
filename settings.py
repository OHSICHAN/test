# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import DeepInteract
import wandb
import random


def get_device(verbose=False):
    d = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if verbose:
        print(f"Using device: {d}")
    return d

# SETTINGS
PROJECT_NAME = "DeepInteract"
WANDB = True
BATCH_SIZE = 200
EVALUATION_SIZE = 510  # 910
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
CUTOFF = None  # 7985
PATIENCE = 12
LOAD_MODEL = (
    None  # "/home/bioscience/dev/DeepInteract/deepinteratct2/devoted-flower-16.pth"
)
