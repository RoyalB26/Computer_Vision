"""Seed management for full experiment reproducibility."""
import os
import random
import numpy as np
import torch


def set_random_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seed across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures cuDNN to use deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
