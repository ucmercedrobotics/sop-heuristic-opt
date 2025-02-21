from typing import Callable
import torch
from torch import Tensor

from sop.utils.graph import TorchGraph
from sop.utils.path import Path
from sop.inference.rollout import RolloutOutput, rollout, categorical_action_selection


# -- Reinforce based losses
def reinforce_loss(scores: Tensor, log_probs: Tensor):
    A = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-9)
    sum_probs = log_probs.sum(-1)
    loss = (A * sum_probs).sum(-1)
    loss = loss.mean(-1)

    return -loss


def reinforce_loss_ER(scores: Tensor, log_probs: Tensor, entropy_coef: float):
    A = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-9)
    sum_probs = log_probs.sum(-1)
    reinforce_loss = (A * sum_probs).sum(-1)

    entropy_loss = -(log_probs.exp() * log_probs).sum(-1).sum(-1)
    loss = reinforce_loss.mean(-1) + entropy_coef * entropy_loss.mean(-1)

    return -loss


# -- Wrapper around base rollout
def heuristic_walk(
    graphs: TorchGraph,
    heuristic: Tensor,
    num_rollouts: int,
    p_f: Tensor,
    action_selection_fn: Callable = categorical_action_selection,
) -> RolloutOutput:
    start_node = graphs.extra["start_node"]
    budget = graphs.extra["budget"]
    goal_node = graphs.extra["goal_node"]

    batch_size, num_nodes = graphs.size()
    indices = torch.arange(batch_size)

    # Loop State
    current_node = start_node.clone()
    current_budget = budget.clone()

    # Initialize Path
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)
    # Mask goal node
    path.mask[indices, goal_node] = 1

    # Rollout
    output = rollout(
        heuristic,
        graphs,
        current_node,
        current_budget,
        path,
        num_rollouts,
        p_f,
        action_selection_fn,
        store_log_probs=True,
    )

    return output
