"""Seed utilities for reproducibility."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set seeds for python, numpy, torch (cpu+cuda), and CuDNN flags for determinism.

    Note: full determinism on GPU requires CUBLAS_WORKSPACE_CONFIG=:4096:8 env var
    (set externally) and benchmark=False. We set deterministic algorithms + benchmark=False;
    for full reproducibility also: torch.use_deterministic_algorithms(True).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
