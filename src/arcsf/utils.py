import os
import random

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))


def seed_everything(seed: int) -> None:
    """Set random seeds for torch, numpy, random, and python.

    Args:
        seed: Seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
