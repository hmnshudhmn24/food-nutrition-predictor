import os
import json
import torch
import random
import numpy as np
from loguru import logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Saved checkpoint to {path}")

def load_classes(labels_file):
    with open(labels_file, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes
