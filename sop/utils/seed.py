import random
import numpy as np
import torch


def random_seed() -> int:
    return random.randint(0, 2**32 - 1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
