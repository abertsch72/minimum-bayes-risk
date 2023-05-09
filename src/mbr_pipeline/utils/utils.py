import numpy as np
import torch
import random

def set_seed(seed):
    if not isinstance(seed, int):
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(warn_only=True)
