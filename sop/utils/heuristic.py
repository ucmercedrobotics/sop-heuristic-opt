from typing import ClassVar
from dataclasses import dataclass
import torch


@dataclass
class SAAHeuristic:
    scores: torch.Tensor
    samples: torch.Tensor


def compute_heuristic(
    rewards: torch.Tensor, weights: torch.Tensor, num_samples: int, kappa: float = 0.5
):
    scores, sampled_costs = sample_average_score(rewards, weights, num_samples, kappa)
    return SAAHeuristic(scores=scores, samples=sampled_costs)


# -- Sample and average cost (heuristic)
def sample_exponential_distribution(rate: torch.Tensor, num_samples: int):
    """Faster Exponential distribution w/ https://en.wikipedia.org/wiki/Inverse_transform_sampling.
    x = -(1/rate)*ln(y)
    """
    sample_shape = (*rate.shape, num_samples)
    y = torch.rand(sample_shape)
    samples = -(1 / rate).unsqueeze(-1) * torch.log(y)
    return samples


def sample_average_score(
    rewards: torch.Tensor, weights: torch.Tensor, num_samples: int, kappa: float = 0.5
):
    rate = 1 / ((1 - kappa) * weights)
    samples = sample_exponential_distribution(rate, num_samples)
    sampled_costs = (kappa * weights).unsqueeze(-1) + samples
    average_cost = sampled_costs.mean(dim=-1)
    score = rewards.unsqueeze(-1) / average_cost
    return score, sampled_costs


if __name__ == "__main__":
    import time

    B = 256  # batch_size
    N = 100  # num_nodes
    M = 100  # num_samples

    dists = torch.ones(size=(B, N, N))
    rewards = torch.ones(size=(B, N))

    start = time.time()
    H = compute_heuristic(rewards, dists, M)
    score_time = time.time() - start
    print(f"Time elapsed heuristic: {score_time}")
