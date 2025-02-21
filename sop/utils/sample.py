import torch
from torch import Tensor
import random
import numpy as np
from datetime import datetime


def random_seed() -> int:
    return int(datetime.now().timestamp())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def sample_range(batch_size: int, min_val: float, max_val: float) -> Tensor:
    return min_val + (max_val - min_val) * torch.rand(batch_size)


def sample_exponential_distribution(rate: Tensor, num_samples: int) -> Tensor:
    """Faster Exponential distribution w/ https://en.wikipedia.org/wiki/Inverse_transform_sampling.
    x = -(1/rate)*ln(y)
    """
    sample_shape = (*rate.shape, num_samples)
    y = torch.rand(sample_shape) + 1e-9
    samples = -(1 / rate).unsqueeze(-1) * torch.log(y)
    return samples


def sample_costs(weights: Tensor, num_samples: int, kappa: float = 0.5) -> Tensor:
    rate = 1 / ((1 - kappa) * weights)
    samples = sample_exponential_distribution(rate, num_samples)
    sampled_costs = (kappa * weights).unsqueeze(-1) + samples
    return sampled_costs
