"""Parameter counting and repeatable benchmark seeds."""

import random
import numpy as np
import torch


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
