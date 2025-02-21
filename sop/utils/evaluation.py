import torch
from torch import Tensor

from sop.utils.graph import TorchGraph
from sop.utils.path import Path
from sop.utils.sample import sample_costs


def evaluate_paths(
    graph: TorchGraph, path: Path, num_samples: int, kappa: float
) -> Tensor:
    # Ensure path is of shape (B, num_rollouts)
    batch_size, num_rollouts, max_length = path.nodes.shape

    b_indices = (
        torch.arange(batch_size).unsqueeze(-1).expand((-1, num_rollouts)).flatten()
    )
    s_indices = torch.arange(batch_size * num_rollouts)

    path = path.flatten()
    total_sampled_cost = torch.zeros((batch_size * num_rollouts, num_samples))

    path_index = 1
    while path_index < max_length:
        prev_node = path.nodes[s_indices, path_index - 1]
        current_node = path.nodes[s_indices, path_index]
        weight = graph.edges["distance"][b_indices, prev_node, current_node]

        is_continuing = current_node != -1
        b_indices, s_indices = b_indices[is_continuing], s_indices[is_continuing]
        if s_indices.numel() == 0:
            break

        sampled_cost = sample_costs(weight[is_continuing], num_samples, kappa)
        total_sampled_cost[s_indices] += sampled_cost
        path_index += 1

    # Reshape
    total_sampled_cost = total_sampled_cost.reshape(
        batch_size, num_rollouts, num_samples
    )

    # Compute average cost
    avg_cost = total_sampled_cost.mean(-1)

    # Compute failure prob
    budget = graph.extra["budget"].unsqueeze(-1).unsqueeze(-1)
    residual = budget - total_sampled_cost
    failure_prob = (residual < 0).sum(-1) / num_samples

    return avg_cost, failure_prob
