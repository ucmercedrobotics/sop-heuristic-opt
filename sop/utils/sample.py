import torch
from torch import Tensor
import time


def sample_exponential_distribution(rate: Tensor, num_samples: int) -> Tensor:
    """Faster Exponential distribution w/ https://en.wikipedia.org/wiki/Inverse_transform_sampling.
    x = -(1/rate)*ln(y)
    """
    sample_shape = (*rate.shape, num_samples)
    y = torch.rand(sample_shape)
    samples = -(1 / rate).unsqueeze(-1) * torch.log(y)
    return samples


def sample_costs(weights: Tensor, num_samples: int, kappa: float = 0.5) -> Tensor:
    rate = 1 / ((1 - kappa) * weights)
    samples = sample_exponential_distribution(rate, num_samples)
    sampled_costs = (kappa * weights).unsqueeze(-1) + samples
    return sampled_costs


if __name__ == "__main__":
    import time

    B = 1024  # batch_size
    N = 100  # num_nodes
    M = 100  # num_samples

    weights = torch.ones(size=(B, N, N))

    _ = sample_costs(weights, M)

    start = time.time()
    sampled_costs = sample_costs(weights, M)
    score_time = time.time() - start
    print(f"Time elapsed heuristic: {score_time}")
